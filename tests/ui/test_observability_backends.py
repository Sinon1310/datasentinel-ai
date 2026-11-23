"""Tests for the observability UI contracts, the demo backend and the not-connected backend."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.observability.contracts import (
    TERMINAL_STATUSES,
    ObservabilityCommands,
    ObservabilityDataSource,
    RunFilters,
    RunStatus,
)
from app.observability.demo import DemoBackend
from app.observability.unavailable import UnavailableBackend

ANCHOR = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)


@pytest.fixture
def demo() -> DemoBackend:
    return DemoBackend(seed=7, anchor=ANCHOR)


def _all(backend, **filters):
    return list(backend.list_runs(RunFilters(**filters), limit=500).items)


class TestContracts:
    def test_both_backends_implement_read_and_command_protocols(self, demo) -> None:
        for backend in (demo, UnavailableBackend()):
            assert isinstance(backend, ObservabilityDataSource)
            assert isinstance(backend, ObservabilityCommands)

    def test_unavailable_backend_is_empty_and_refuses_actions(self) -> None:
        backend = UnavailableBackend()
        assert backend.is_connected is False and backend.kind == "live"
        summary = backend.summary(RunFilters())
        assert summary.total_runs == 0 and summary.success_rate_pct is None and summary.avg_duration_ms is None
        assert backend.list_runs(RunFilters()).total == 0 and backend.get_run("x") is None
        assert not backend.request_retry("x", "a", "b").ok and not backend.cancel_pending("x", "a", "b").ok


class TestDemoIsolationAndDeterminism:
    def test_deterministic_for_seed_and_anchor(self) -> None:
        a, b = DemoBackend(seed=7, anchor=ANCHOR), DemoBackend(seed=7, anchor=ANCHOR)
        assert [r.pipeline_run_id for r in _all(a)] == [r.pipeline_run_id for r in _all(b)]
        assert [r.pipeline_run_id for r in _all(DemoBackend(seed=8, anchor=ANCHOR))] != [r.pipeline_run_id for r in _all(a)]

    def test_every_record_is_labelled_synthetic(self, demo) -> None:
        runs = _all(demo)
        assert runs and all(r.is_synthetic and r.pipeline_id.startswith("demo.") for r in runs)
        assert all(p.is_synthetic and p.pipeline_id.startswith("demo.") for p in demo.list_pipelines())
        assert _all(demo, include_synthetic=False) == []  # never presented as real data

    def test_nothing_is_written_to_disk(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        backend = DemoBackend(seed=1, anchor=ANCHOR)
        failed = next(r for r in _all(backend, statuses=(RunStatus.FAILED,))
                      if backend.get_run(r.pipeline_run_id).allowed_actions.can_request_retry)
        backend.request_retry(failed.pipeline_run_id, "alice", "test")
        assert list(tmp_path.iterdir()) == []

    def test_requires_timezone_aware_anchor(self) -> None:
        with pytest.raises(ValueError):
            DemoBackend(anchor=datetime(2026, 1, 1))


class TestDemoAggregates:
    def test_summary_matches_independent_recomputation(self, demo) -> None:
        runs = _all(demo)
        summary = demo.summary(RunFilters())
        terminal = [r for r in runs if r.status in TERMINAL_STATUSES]
        success = sum(1 for r in runs if r.status == RunStatus.SUCCESS)
        durations = sorted(r.duration_ms for r in terminal if r.duration_ms is not None)
        assert summary.total_runs == len(runs)
        assert summary.success_rate_pct == pytest.approx(success / len(terminal) * 100)
        assert summary.avg_duration_ms == pytest.approx(sum(durations) / len(durations))
        assert summary.p95_duration_ms == durations[math.ceil(0.95 * len(durations)) - 1]
        assert summary.total_input_rows == sum(r.input_rows for r in runs if r.input_rows is not None)
        assert summary.running_count == 2 and summary.stale_count == 1 and summary.pending_count == 1

    def test_filters(self, demo) -> None:
        pipeline = "demo.orders_daily_load"
        assert {r.pipeline_id for r in _all(demo, pipeline_ids=(pipeline,))} == {pipeline}
        assert {r.status for r in _all(demo, statuses=(RunStatus.FAILED,))} == {RunStatus.FAILED}
        assert {r.trigger_type for r in _all(demo, trigger_types=("retry",))} == {"retry"}
        recent = _all(demo, start=ANCHOR - timedelta(days=1), end=ANCHOR + timedelta(hours=1))
        assert recent and all(r.created_at >= ANCHOR - timedelta(days=1) for r in recent)
        stale = _all(demo, stale_only=True)
        assert len(stale) == 1 and stale[0].status == RunStatus.RUNNING and stale[0].display_status == "RUNNING · STALE"

    def test_stale_is_derived_not_persisted(self, demo) -> None:
        stale = _all(demo, stale_only=True)[0]
        detail = demo.get_run(stale.pipeline_run_id)
        assert detail.run.status == RunStatus.RUNNING and detail.run.is_stale
        assert ANCHOR - detail.heartbeat_at > demo.stale_after

    def test_trends_and_health_cover_the_same_runs(self, demo) -> None:
        total = sum(p.success + p.failed + p.cancelled + p.active for p in demo.trends(RunFilters()))
        assert total == demo.summary(RunFilters()).total_runs
        assert sum(h.runs for h in demo.pipeline_health(RunFilters())) == total
        assert sum(f.count for f in demo.failure_distribution(RunFilters())) == sum(
            1 for r in _all(demo) if r.failure_category is not None)


class TestDemoCommands:
    def _latest_failed(self, demo):
        for run in _all(demo, statuses=(RunStatus.FAILED,)):
            if demo.get_run(run.pipeline_run_id).allowed_actions.can_request_retry:
                return run
        raise AssertionError("no retryable run")

    def test_retry_creates_linked_pending_attempt_and_executes_nothing(self, demo) -> None:
        run = self._latest_failed(demo)
        before = {r.pipeline_run_id: r.status for r in _all(demo)}
        result = demo.request_retry(run.pipeline_run_id, "alice", "upstream fixed")
        assert result.ok and "Nothing was executed" in result.message
        new = demo.get_run(result.created_pipeline_run_id)
        assert new.run.status == RunStatus.PENDING and new.run.trigger_type == "retry"
        assert new.run.attempt_number == run.attempt_number + 1 and new.requested_by == "alice"
        assert new.root_pipeline_run_id == demo.get_run(run.pipeline_run_id).root_pipeline_run_id
        after = {r.pipeline_run_id: r.status for r in _all(demo)}
        assert {k: v for k, v in after.items() if k in before} == before  # no existing run changed status
        assert not demo.get_run(run.pipeline_run_id).allowed_actions.can_request_retry  # no longer latest
        assert any(e.event_type == "retry_requested" for e in demo.get_run(run.pipeline_run_id).events)

    @pytest.mark.parametrize("status", [RunStatus.SUCCESS, RunStatus.RUNNING, RunStatus.PENDING])
    def test_retry_refused_for_non_failed_runs(self, demo, status) -> None:
        run = _all(demo, statuses=(status,))[0]
        result = demo.request_retry(run.pipeline_run_id, "alice", "x")
        assert not result.ok and status.value in result.message

    def test_retry_refused_for_superseded_attempt(self, demo) -> None:
        superseded = next(r for r in _all(demo, statuses=(RunStatus.FAILED,))
                          if not demo.get_run(r.pipeline_run_id).allowed_actions.can_request_retry)
        assert "latest attempt" in demo.request_retry(superseded.pipeline_run_id, "a", "b").message

    def test_cancel_only_pending(self, demo) -> None:
        running = _all(demo, statuses=(RunStatus.RUNNING,))[0]
        assert not demo.cancel_pending(running.pipeline_run_id, "alice", "x").ok
        pending = _all(demo, statuses=(RunStatus.PENDING,))[0]
        result = demo.cancel_pending(pending.pipeline_run_id, "alice", "not needed")
        assert result.ok and demo.get_run(pending.pipeline_run_id).run.status == RunStatus.CANCELLED
        assert not demo.cancel_pending(pending.pipeline_run_id, "alice", "again").ok  # now terminal

    @pytest.mark.parametrize("who, why", [("", "reason"), ("alice", " ")])
    def test_actions_require_actor_and_reason(self, demo, who, why) -> None:
        pending = _all(demo, statuses=(RunStatus.PENDING,))[0]
        assert not demo.cancel_pending(pending.pipeline_run_id, who, why).ok
        assert not demo.request_retry(self._latest_failed(demo).pipeline_run_id, who, why).ok
