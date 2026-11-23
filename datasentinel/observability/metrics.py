"""Deterministic metric formulas (pure functions; no I/O, no clocks, no randomness).

Per-run metrics (written once by the store inside the terminal transaction):

* ``duration_ms``: in-process monotonic elapsed time when captured; else ``ended - started`` wall clock
  when both exist and the end is not before the start; else unavailable (never guessed).
* ``input_throughput_rows_per_s``: ``input_rows / (duration_ms / 1000)``; unavailable without rows or
  without a positive duration.
* ``task_attempt_count`` (task rows), ``failed_task_attempt_count`` (FAILED task rows),
  ``failed_task_count`` (distinct tasks whose FINAL attempt FAILED) and ``task_retry_count`` (task
  attempts beyond each task's first).

Aggregates over many runs:

* success rate = SUCCESS ÷ (SUCCESS + FAILED + CANCELLED); unavailable when no run is terminal.
* percentiles use the nearest-rank method in Python (not SQLite's non-portable ``percentile()``).
* STALE is derived from heartbeats (see :func:`~datasentinel.observability.lifecycle.is_stale`).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from datasentinel.observability import lifecycle
from datasentinel.observability.models import (
    DurationSource, PipelineRun, PipelineRunMetrics, PipelineRunStatus, PipelineTaskRun, TaskStatus,
)

METRICS_VERSION = "v1"


def derive_run_metrics(run: PipelineRun, tasks: Sequence[PipelineTaskRun], computed_at_ms: int) -> PipelineRunMetrics:
    """Derive the stored metrics of a terminal run from its persisted facts."""
    if run.monotonic_elapsed_ms is not None:
        duration: Optional[int] = run.monotonic_elapsed_ms
        source = DurationSource.MONOTONIC
    elif (run.started_at_ms is not None and run.ended_at_ms is not None
          and run.ended_at_ms >= run.started_at_ms):
        duration, source = run.ended_at_ms - run.started_at_ms, DurationSource.WALL_CLOCK
    else:
        duration, source = None, DurationSource.UNAVAILABLE
    return PipelineRunMetrics(
        pipeline_run_id=run.pipeline_run_id,
        metrics_version=METRICS_VERSION,
        duration_ms=duration,
        duration_source=source,
        input_rows=run.input_rows,
        output_rows=run.output_rows,
        rows_source=run.rows_source,
        input_throughput_rows_per_s=throughput_rows_per_s(run.input_rows, duration),
        task_attempt_count=len(tasks),
        failed_task_attempt_count=sum(1 for t in tasks if t.status is TaskStatus.FAILED),
        failed_task_count=failed_task_count(tasks),
        task_retry_count=task_retry_count(tasks),
        computed_at_ms=computed_at_ms,
    )


def throughput_rows_per_s(rows: Optional[int], duration_ms: Optional[int]) -> Optional[float]:
    """Rows per second, or ``None`` when rows are unknown or the duration is unknown or zero."""
    if rows is None or not duration_ms or duration_ms <= 0:
        return None
    return round(rows / (duration_ms / 1000), 6)


def _final_attempts(tasks: Sequence[PipelineTaskRun]) -> Dict[str, PipelineTaskRun]:
    final: Dict[str, PipelineTaskRun] = {}
    for task in tasks:
        if task.task_name not in final or task.task_attempt > final[task.task_name].task_attempt:
            final[task.task_name] = task
    return final


def failed_task_count(tasks: Sequence[PipelineTaskRun]) -> int:
    """Distinct tasks whose final attempt FAILED."""
    return sum(1 for t in _final_attempts(tasks).values() if t.status is TaskStatus.FAILED)


def task_retry_count(tasks: Sequence[PipelineTaskRun]) -> int:
    """Task attempts beyond each task's first attempt."""
    return len(tasks) - len(_final_attempts(tasks))


def nearest_rank_percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    """Nearest-rank percentile (``0 < percentile <= 100``) of ``values``; ``None`` for no values.

    The result is always one of the observed values: rank = ceil(p/100 × n) in the sorted list.
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    if not ordered:
        return None
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def success_rate(statuses: Iterable[PipelineRunStatus]) -> Optional[float]:
    """SUCCESS ÷ terminal runs, as a fraction in [0, 1]; ``None`` when no run is terminal."""
    counts = Counter(PipelineRunStatus(s) for s in statuses)
    terminal = sum(counts[s] for s in lifecycle.TERMINAL_RUN_STATUSES)
    return None if terminal == 0 else counts[PipelineRunStatus.SUCCESS] / terminal


@dataclass(frozen=True)
class DurationStats:
    """Duration statistics over runs whose duration is available."""

    count: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    min_ms: Optional[int]
    max_ms: Optional[int]
    by_source: Mapping[str, int]


def duration_stats(metrics: Iterable[PipelineRunMetrics]) -> DurationStats:
    """p50/p95/min/max over available durations, with a count of each duration source."""
    items = list(metrics)
    durations = [m.duration_ms for m in items if m.duration_ms is not None]
    by_source = Counter(m.duration_source.value for m in items)
    return DurationStats(
        count=len(durations),
        p50_ms=nearest_rank_percentile(durations, 50),
        p95_ms=nearest_rank_percentile(durations, 95),
        min_ms=min(durations) if durations else None,
        max_ms=max(durations) if durations else None,
        by_source={s.value: by_source.get(s.value, 0) for s in DurationSource},
    )


@dataclass(frozen=True)
class RunAggregate:
    """Deterministic summary of a set of pipeline runs."""

    total: int
    by_status: Mapping[str, int]
    terminal: int
    success_rate: Optional[float]
    stale: int
    duration: DurationStats
    task_retry_count: int
    failed_task_count: int


def aggregate_runs(runs: Sequence[PipelineRun], metrics_by_run: Mapping[str, PipelineRunMetrics], *,
                   now_ms: int, stale_after_ms: int) -> RunAggregate:
    """Summarise runs. Metrics exist only for terminal runs; STALE is derived, never read from storage."""
    by_status = Counter(r.status.value for r in runs)
    metrics = [metrics_by_run[r.pipeline_run_id] for r in runs if r.pipeline_run_id in metrics_by_run]
    stale = sum(1 for r in runs if lifecycle.is_stale(r.status, heartbeat_at_ms=r.heartbeat_at_ms,
                                                      started_at_ms=r.started_at_ms, now_ms=now_ms,
                                                      stale_after_ms=stale_after_ms))
    return RunAggregate(
        total=len(runs),
        by_status={s.value: by_status.get(s.value, 0) for s in PipelineRunStatus},
        terminal=sum(by_status.get(s.value, 0) for s in lifecycle.TERMINAL_RUN_STATUSES),
        success_rate=success_rate(r.status for r in runs),
        stale=stale,
        duration=duration_stats(metrics),
        task_retry_count=sum(m.task_retry_count for m in metrics),
        failed_task_count=sum(m.failed_task_count for m in metrics),
    )
