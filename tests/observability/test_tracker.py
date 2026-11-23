"""PipelineTracker: run/task tracking, failure recording, retries (record-only), claims, cancellation, heartbeats."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from datasentinel.config import Settings
from datasentinel.exceptions import QualityGateError, TransformationError
from datasentinel.observability import (
    DurationSource, EventType, FailureCategory, FixedClock, InvalidTransitionError, ObservabilityStore,
    ObservabilityValidationError, PipelineRunStatus, PipelineTracker, RetryNotAllowedError, RowsSource, TaskStatus,
    TerminalStateError, TriggerType,
)
from datasentinel.observability.gates import enforce_quality_gate
from datasentinel.quality.models import QualityReport

R = PipelineRunStatus


@pytest.fixture
def tracker(store: ObservabilityStore, clock: FixedClock) -> PipelineTracker:
    return PipelineTracker(store, clock=clock, heartbeat_min_interval_s=15)


def _run_id_of(tracker: PipelineTracker, **kwargs) -> str:
    with tracker.track_run("orders.daily", **kwargs) as run:
        return run.pipeline_run_id


# ------------------------------------------------------------------------------------------ success

def test_successful_run_with_tasks_rows_and_monotonic_durations(tracker, store, clock) -> None:
    with tracker.track_run("orders.daily", display_name="Orders daily", parameters={"date": "2026-11-23"},
                           code_version="v1.2.3", actor="scheduler") as run:
        with run.task("extract") as task:
            clock.advance(seconds=2)
            task.record_rows(output_rows=1_000, source=RowsSource.MEASURED)
        with run.task("optional_step") as task:
            task.skip()
        run.record_rows(input_rows=1_000, output_rows=990, source=RowsSource.MEASURED)
        clock.advance(seconds=1)
    stored = store.get_run(run.pipeline_run_id)
    assert stored.status is R.SUCCESS and stored.status_changed_by == "scheduler"
    assert stored.parameters == {"date": "2026-11-23"} and stored.code_version == "v1.2.3"
    tasks = {t.task_name: t for t in store.get_tasks(run.pipeline_run_id)}
    assert tasks["extract"].status is TaskStatus.SUCCESS and tasks["extract"].monotonic_elapsed_ms == 2_000
    assert tasks["extract"].output_rows == 1_000 and tasks["optional_step"].status is TaskStatus.SKIPPED
    m = store.get_metrics(run.pipeline_run_id)
    assert (m.duration_ms, m.duration_source) == (3_000, DurationSource.MONOTONIC)
    assert m.input_throughput_rows_per_s == round(1_000 / 3, 6)
    assert store.get_failures(run.pipeline_run_id) == []
    assert run.closed
    with pytest.raises(TerminalStateError):
        run.heartbeat(force=True)


def test_default_actor_and_display_name(tracker, store) -> None:
    rid = _run_id_of(tracker)
    assert store.get_run(rid).status_changed_by == "pipeline:orders.daily"
    assert store.get_pipeline("orders.daily").display_name == "orders.daily"


# ------------------------------------------------------------------------------------------ failure

def test_exception_fails_the_run_reraises_and_links_the_failing_task(tracker, store) -> None:
    with pytest.raises(TransformationError, match="cast failed"):
        with tracker.track_run("orders.daily") as run:
            with run.task("transform"):
                raise TransformationError("cast failed for column amount")
    rid = run.pipeline_run_id
    assert store.get_run(rid).status is R.FAILED
    # one primary failure (what ended the run) + the durable record of the failed task attempt
    failures = store.get_failures(rid)
    assert [f.is_primary for f in failures] == [True, False]
    failure, task_record = failures
    assert task_record.pipeline_task_run_id == failure.pipeline_task_run_id and task_record.stage == "transform"
    assert failure.category is FailureCategory.TRANSFORMATION
    assert failure.stage == "transform" and failure.pipeline_task_run_id == store.get_tasks(rid)[0].pipeline_task_run_id
    assert failure.classifier_rule == "type.transformation" and failure.classifier_version == "v1"
    assert failure.context["classification"]["matched_link"] == 0
    assert all(set(frame) == {"module", "function", "lineno"} for frame in failure.stack)
    assert store.get_tasks(rid)[0].status is TaskStatus.FAILED


def test_wrapped_task_exception_is_still_one_primary_failure(tracker, store) -> None:
    with pytest.raises(RuntimeError):
        with tracker.track_run("orders.daily") as run:
            try:
                with run.task("load"):
                    raise OSError("disk full")
            except OSError as exc:
                raise RuntimeError("pipeline aborted") from exc
    failures = store.get_failures(run.pipeline_run_id)
    assert sum(f.is_primary for f in failures) == 1
    failure = failures[0]
    assert failure.is_primary and failure.stage == "load" and failure.category is FailureCategory.RESOURCE_RUNTIME
    assert failures[1].error_type == "builtins.OSError" and not failures[1].is_primary
    assert failure.error_type == "builtins.RuntimeError" and failure.exception_chain[0]["type"] == "builtins.OSError"


def test_retried_task_failure_is_kept_as_non_primary_on_success(tracker, store) -> None:
    with tracker.track_run("orders.daily") as run:
        for attempt in range(3):
            try:
                with run.task("fetch"):
                    if attempt < 2:
                        raise TimeoutError(f"attempt {attempt}")
                break
            except TimeoutError:
                continue
    rid = run.pipeline_run_id
    assert store.get_run(rid).status is R.SUCCESS
    failures = store.get_failures(rid)
    assert len(failures) == 2 and not any(f.is_primary for f in failures)
    assert [t.task_attempt for t in store.get_tasks(rid)] == [1, 2, 3]
    m = store.get_metrics(rid)
    assert (m.task_retry_count, m.failed_task_attempt_count, m.failed_task_count) == (2, 2, 0)


def test_unrelated_run_failure_keeps_earlier_task_failures_non_primary(tracker, store) -> None:
    with pytest.raises(ValueError):
        with tracker.track_run("orders.daily") as run:
            try:
                with run.task("optional"):
                    raise KeyError("missing")
            except KeyError:
                pass
            raise ValueError("final step failed")
    failures = store.get_failures(run.pipeline_run_id)
    primary = [f for f in failures if f.is_primary]
    assert len(primary) == 1 and primary[0].error_type == "builtins.ValueError" and primary[0].stage is None
    assert [f.error_type for f in failures if not f.is_primary] == ["builtins.KeyError"]


def test_quality_gate_failure_is_data_quality_with_gate_context(tracker, store) -> None:
    report = QualityReport("d.csv", "t", 41.5, "F", [], [])
    with pytest.raises(QualityGateError):
        with tracker.track_run("orders.daily") as run:
            with run.task("validate"):
                enforce_quality_gate(report)
    failure = [f for f in store.get_failures(run.pipeline_run_id) if f.is_primary][0]
    assert failure.category is FailureCategory.DATA_QUALITY
    assert failure.context["quality_gate"]["overall_score"] == 41.5
    assert failure.context["quality_gate"]["reasons"] == ["quality score 41.50 is below the minimum 60.00"]


def test_reported_category_via_exception_attribute_or_explicit_fail(tracker, store) -> None:
    class VendorError(Exception):
        failure_category = FailureCategory.INGESTION

    with pytest.raises(VendorError):
        with tracker.track_run("orders.daily") as run:
            raise VendorError("vendor feed missing")
    failure = store.get_failures(run.pipeline_run_id)[0]
    assert (failure.category, failure.category_source.value) == (FailureCategory.INGESTION, "reported")

    handle = tracker.start_run("orders.daily")
    handle.fail(category=FailureCategory.SCHEMA, message="contract drift detected for jane@corp.com")
    failure = store.get_failures(handle.pipeline_run_id)[0]
    assert failure.category is FailureCategory.SCHEMA and "jane@corp.com" not in failure.error_message
    with pytest.raises(ObservabilityValidationError):
        tracker.start_run("orders.daily").fail()


def test_interrupts_cancel_the_run(tracker, store) -> None:
    with pytest.raises(KeyboardInterrupt):
        with tracker.track_run("orders.daily") as run:
            with run.task("long"):
                raise KeyboardInterrupt
    rid = run.pipeline_run_id
    assert store.get_run(rid).status is R.CANCELLED
    assert store.get_tasks(rid)[0].status is TaskStatus.CANCELLED
    assert store.get_failures(rid) == []
    note = [e for e in store.get_events(rid) if e.event_type is EventType.NOTE][0]
    assert note.details == {"cancel_reason": "interrupted (KeyboardInterrupt)"}


def test_tracking_errors_never_mask_the_pipeline_exception(tracker, store, db_path: Path) -> None:
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute("CREATE TRIGGER inject BEFORE INSERT ON pipeline_failures BEGIN SELECT RAISE(ABORT, 'x'); END")
    try:
        with pytest.raises(ValueError, match="the real problem") as info:
            with tracker.track_run("orders.daily") as run:
                raise ValueError("the real problem")
    finally:
        raw.execute("DROP TRIGGER inject")
        raw.close()
    assert any("could not record" in note for note in info.value.__notes__)
    assert store.get_run(run.pipeline_run_id).status is R.RUNNING  # honestly left open (shown as stale later)


def test_explicit_owner_cancel(tracker, store) -> None:
    with tracker.track_run("orders.daily") as run:
        run.cancel("operator stopped the backfill")
    assert store.get_run(run.pipeline_run_id).status is R.CANCELLED


# ---------------------------------------------------------------------------------- retries & claims

def _failed(tracker) -> str:
    with pytest.raises(ValueError):
        with tracker.track_run("orders.daily") as run:
            raise ValueError("boom")
    return run.pipeline_run_id


def test_retry_request_is_record_only(tracker, store) -> None:
    executed = []

    def pipeline() -> None:  # the real pipeline: must never be invoked by DataSentinel
        executed.append(True)

    first = _failed(tracker)
    retry = tracker.request_retry(first, requested_by="jane.doe@example.com", reason="upstream fixed (ticket 123456789)")
    assert executed == [] and retry.status is R.PENDING and retry.attempt_number == 2
    assert retry.requested_by == "<email>" and retry.retry_reason == "upstream fixed (ticket <number>)"
    assert store.get_tasks(retry.pipeline_run_id) == [] and retry.started_at_ms is None
    with pytest.raises(ObservabilityValidationError):
        tracker.request_retry(retry.pipeline_run_id, requested_by="x", reason="  ")
    with pytest.raises(RetryNotAllowedError):
        tracker.request_retry(retry.pipeline_run_id, requested_by="x", reason="still pending")
    assert pipeline is not None and executed == []


def test_pending_retry_is_claimed_only_by_the_pipeline(tracker, store) -> None:
    retry = tracker.request_retry(_failed(tracker), requested_by="ops", reason="retry")
    with pytest.raises(ObservabilityValidationError, match="different pipeline"):
        with tracker.track_run("other.pipeline", attach_to=retry.pipeline_run_id):
            pass
    with tracker.track_run("orders.daily", attach_to=retry.pipeline_run_id) as run:
        with run.task("extract"):
            pass
    assert run.pipeline_run_id == retry.pipeline_run_id and run.attempt_number == 2
    assert store.get_run(retry.pipeline_run_id).status is R.SUCCESS
    assert [a.status for a in store.get_attempt_chain(retry.pipeline_run_id)] == [R.FAILED, R.SUCCESS]


def test_only_one_process_can_claim_a_pending_run(tracker, store) -> None:
    pending = tracker.create_pending_run("orders.daily", trigger=TriggerType.SCHEDULE)
    barrier, won, lost = threading.Barrier(6), [], []

    def claim() -> None:
        barrier.wait()
        try:
            won.append(tracker.claim_pending_run(pending.pipeline_run_id))
        except InvalidTransitionError as exc:
            lost.append(exc)

    threads = [threading.Thread(target=claim) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1 and len(lost) == 5


def test_dashboard_cancels_only_pending_runs(tracker, store) -> None:
    pending = tracker.create_pending_run("orders.daily")
    cancelled = tracker.cancel_pending_run(pending.pipeline_run_id, actor="dashboard user", reason="not needed")
    assert cancelled.status is R.CANCELLED
    running = tracker.start_run("orders.daily")
    with pytest.raises(InvalidTransitionError, match="owns it"):
        tracker.cancel_pending_run(running.pipeline_run_id, actor="dashboard user")
    assert store.get_run(running.pipeline_run_id).status is R.RUNNING
    with pytest.raises(TerminalStateError):
        tracker.cancel_pending_run(pending.pipeline_run_id, actor="dashboard user")


def test_cancelled_pending_retry_can_be_retried_again(tracker, store) -> None:
    first = _failed(tracker)
    second = tracker.request_retry(first, requested_by="ops", reason="r1")
    tracker.cancel_pending_run(second.pipeline_run_id, actor="ops", reason="abandoned")
    third = tracker.request_retry(second.pipeline_run_id, requested_by="ops", reason="r2")
    assert [a.attempt_number for a in store.get_attempt_chain(first)] == [1, 2, 3]
    assert third.retry_of_pipeline_run_id == second.pipeline_run_id


# ---------------------------------------------------------------------------------------- heartbeats

def test_heartbeats_are_throttled_by_monotonic_time(tracker, store, clock) -> None:
    handle = tracker.start_run("orders.daily")
    assert handle.heartbeat() is False  # the start counts as the first sign of life
    clock.advance(seconds=14.999)
    assert handle.heartbeat() is False
    clock.advance(ms=1)
    assert handle.heartbeat() is True
    written = store.get_run(handle.pipeline_run_id).heartbeat_at_ms
    clock.advance(seconds=5)
    assert handle.heartbeat() is False and handle.heartbeat(force=True) is True
    assert store.get_run(handle.pipeline_run_id).heartbeat_at_ms == written + 5_000
    assert not [e for e in store.get_events(handle.pipeline_run_id) if e.event_type.value == "heartbeat"]


def test_finishing_a_task_sends_a_throttled_heartbeat(tracker, store, clock) -> None:
    with tracker.track_run("orders.daily") as run:
        with run.task("slow"):
            clock.advance(seconds=20)
        assert store.get_run(run.pipeline_run_id).heartbeat_at_ms == clock.utc_now_ms()


# ---------------------------------------------------------------------------------------- validation

def test_invalid_inputs_are_rejected_before_writing(tracker, store) -> None:
    with pytest.raises(ObservabilityValidationError):
        tracker.start_run("Not A Slug")
    with pytest.raises(ObservabilityValidationError):
        tracker.start_run("orders.daily", trigger=TriggerType.RETRY)
    with pytest.raises(ObservabilityValidationError):
        PipelineTracker(store, heartbeat_min_interval_s=0)
    with pytest.raises(ObservabilityValidationError):
        PipelineTracker("not a store")  # type: ignore[arg-type]


def test_from_settings_uses_configured_database_and_interval(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "d", pipeline_heartbeat_min_interval_seconds=30)
    tracker = PipelineTracker.from_settings(settings, clock=FixedClock())
    assert tracker._store.path == (tmp_path / "d" / "observability.sqlite").absolute()
    assert tracker._heartbeat_interval_ns == 30 * 1_000_000_000
