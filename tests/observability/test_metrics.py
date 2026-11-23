"""Deterministic metric formulas, checked against hand computation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from datasentinel.observability import (
    DurationSource, PipelineRun, PipelineRunMetrics, PipelineRunStatus, PipelineTaskRun, RowsSource, TaskStatus,
    TriggerType,
)
from datasentinel.observability import store as store_module
from datasentinel.observability.metrics import (
    aggregate_runs, derive_run_metrics, duration_stats, failed_task_count, nearest_rank_percentile, success_rate,
    task_retry_count, throughput_rows_per_s,
)

S = PipelineRunStatus


def _run(i: int = 0, status: S = S.SUCCESS, *, started: int | None = 1_000, ended: int | None = 5_000,
         mono: int | None = None, rows: int | None = None, heartbeat: int | None = None) -> PipelineRun:
    rid = f"{i:032x}"
    return PipelineRun(rid, "p", rid, None, 1, status, "a", 0, TriggerType.MANUAL, None, None, 0, started,
                       ended if status in (S.SUCCESS, S.FAILED, S.CANCELLED) else None, mono, heartbeat, rows, None,
                       RowsSource.MEASURED if rows is not None else None)


def _task(name: str, attempt: int, status: TaskStatus) -> PipelineTaskRun:
    return PipelineTaskRun(f"{hash((name, attempt)) & (2**128 - 1):032x}", "0" * 32, name, attempt, status, 0, 1,
                           None, None, None, None)


def test_the_store_uses_the_metrics_module() -> None:
    assert store_module.derive_run_metrics is derive_run_metrics


@pytest.mark.parametrize("kwargs,expected", [
    ({"mono": 1_234}, (1_234, DurationSource.MONOTONIC)),
    ({}, (4_000, DurationSource.WALL_CLOCK)),
    ({"ended": 900}, (None, DurationSource.UNAVAILABLE)),        # end before start (clock moved back)
    ({"started": None}, (None, DurationSource.UNAVAILABLE)),     # never started
    ({"ended": 1_000}, (0, DurationSource.WALL_CLOCK)),
])
def test_duration_sources(kwargs, expected) -> None:
    m = derive_run_metrics(_run(**kwargs), [], 9)
    assert (m.duration_ms, m.duration_source) == expected and m.computed_at_ms == 9


@pytest.mark.parametrize("rows,duration,expected", [(5_000, 2_500, 2_000.0), (None, 2_500, None), (10, 0, None),
                                                    (10, None, None), (1, 3_000, 0.333333), (0, 1_000, 0.0)])
def test_throughput(rows, duration, expected) -> None:
    assert throughput_rows_per_s(rows, duration) == expected


def test_task_counts_use_final_attempts() -> None:
    tasks = [_task("a", 1, TaskStatus.FAILED), _task("a", 2, TaskStatus.SUCCESS),
             _task("b", 1, TaskStatus.FAILED), _task("b", 2, TaskStatus.FAILED), _task("b", 3, TaskStatus.FAILED),
             _task("c", 1, TaskStatus.SKIPPED)]
    assert (failed_task_count(tasks), task_retry_count(tasks)) == (1, 3)
    m = derive_run_metrics(_run(), tasks, 0)
    assert (m.task_attempt_count, m.failed_task_attempt_count, m.failed_task_count, m.task_retry_count) == (6, 4, 1, 3)


@pytest.mark.parametrize("values,p,expected", [
    ([], 50, None), ([7], 95, 7), ([1, 2, 3, 4], 50, 2), ([1, 2, 3, 4], 95, 4), (list(range(1, 101)), 95, 95),
    (list(range(1, 21)), 95, 19), ([5, 1, 3], 100, 5), ([5, 1, 3], 1, 1),
])
def test_nearest_rank_percentile(values, p, expected) -> None:
    assert nearest_rank_percentile(values, p) == expected


@pytest.mark.parametrize("p", [0, -1, 101])
def test_percentile_bounds(p) -> None:
    with pytest.raises(ValueError):
        nearest_rank_percentile([1], p)


def test_success_rate_counts_terminal_runs_only() -> None:
    assert success_rate([S.SUCCESS, S.SUCCESS, S.FAILED, S.CANCELLED, S.RUNNING, S.PENDING]) == 0.5
    assert success_rate([S.RUNNING, S.PENDING]) is None
    assert success_rate([]) is None


def test_duration_stats_and_aggregate() -> None:
    runs = [_run(1, mono=1_000), _run(2, mono=3_000), _run(3, S.FAILED, ended=900), _run(4, S.CANCELLED),
            _run(5, S.RUNNING, heartbeat=10_000), _run(6, S.RUNNING, heartbeat=99_000), _run(7, S.PENDING,
                                                                                            started=None)]
    metrics = {r.pipeline_run_id: replace(derive_run_metrics(r, [], 0), task_retry_count=1)
               for r in runs if r.is_terminal}
    agg = aggregate_runs(runs, metrics, now_ms=100_000, stale_after_ms=60_000)
    assert (agg.total, agg.terminal, agg.success_rate, agg.stale) == (7, 4, 0.5, 1)
    assert agg.by_status == {"PENDING": 1, "RUNNING": 2, "SUCCESS": 2, "FAILED": 1, "CANCELLED": 1}
    d = agg.duration
    assert (d.count, d.p50_ms, d.p95_ms, d.min_ms, d.max_ms) == (3, 3_000, 4_000, 1_000, 4_000)
    assert d.by_source == {"monotonic": 2, "wall_clock": 1, "unavailable": 1}
    assert agg.task_retry_count == 4
    empty = duration_stats([])
    assert (empty.count, empty.p50_ms, empty.min_ms) == (0, None, None)


def test_metrics_are_deterministic() -> None:
    run = _run(mono=777, rows=12_345)
    assert len({derive_run_metrics(run, [], 1) for _ in range(10)}) == 1
    assert isinstance(derive_run_metrics(run, [], 1), PipelineRunMetrics)
