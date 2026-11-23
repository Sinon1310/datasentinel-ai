"""Deterministic, in-memory synthetic pipeline history for demos and visual development.

Isolation guarantees:

* Nothing is written to disk or to any database; data lives only in this object (one per browser
  session in the UI).
* Every pipeline id starts with ``demo.`` and every record has ``is_synthetic=True``.
* Generation is deterministic for a given ``(seed, anchor)``; the UI uses today's UTC hour as anchor.
* Actions follow the approved semantics exactly: a retry request creates a PENDING attempt linked to
  the latest FAILED/CANCELLED attempt (nothing is executed); only PENDING runs can be cancelled.

The aggregates below play the role of the future backend's authoritative queries for demo data.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from app.observability.contracts import (
    TERMINAL_STATUSES,
    ActionResult,
    AllowedActions,
    AttemptView,
    FailureCategory,
    FailureStat,
    FailureTypeStat,
    FailureView,
    ObservabilitySummary,
    PipelineHealth,
    PipelineInfo,
    RunDetail,
    RunEventView,
    RunFilters,
    RunMetricsView,
    RunPage,
    RunStatus,
    RunSummary,
    StatusCount,
    TaskRunView,
    TaskStatus,
    TrendPoint,
)

_NAMESPACE = uuid.UUID("6f1d4c2e-8a3b-4f5e-9c7d-1a2b3c4d5e6f")
DEMO_ACTOR = "demo.operator"
DEMO_NOTICE = "Demo only: recorded in this browser session. Nothing was executed or persisted."


@dataclass(frozen=True)
class _PipelineSpec:
    pipeline_id: str
    name: str
    description: str
    owner: str
    tasks: tuple[str, ...]
    cadence_hours: int
    base_duration_ms: int
    rows: tuple[int, int]
    failure_rate: float
    failures: tuple[tuple[FailureCategory, str, str, str], ...]  # category, error type, message, failing task


_SPECS: tuple[_PipelineSpec, ...] = (
    _PipelineSpec("demo.orders_daily_load", "Orders daily load", "Loads daily order extracts into the warehouse.",
                  "data-platform", ("extract_orders", "validate_schema", "transform_orders", "load_warehouse"),
                  24, 420_000, (180_000, 260_000), 0.14, (
                      (FailureCategory.SCHEMA, "datasentinel.exceptions.SchemaContractError",
                       "Required column 'order_channel' is missing from the extract", "validate_schema"),
                      (FailureCategory.DATA_QUALITY, "datasentinel.exceptions.QualityGateError",
                       "Quality gate failed: critical check uniqueness.duplicate_records (score 54.2 < 60)",
                       "validate_schema"),
                  )),
    _PipelineSpec("demo.customer_dim_refresh", "Customer dimension refresh", "SCD2 merge of CRM customers.",
                  "analytics-eng", ("extract_crm", "dedupe_customers", "scd2_merge"), 12, 180_000, (40_000, 60_000),
                  0.08, (
                      (FailureCategory.INGESTION, "datasentinel.exceptions.DatasetParseError",
                       "Malformed JSON in CRM export (line 1842, column 17)", "extract_crm"),
                      (FailureCategory.TRANSFORMATION, "datasentinel.exceptions.TransformationError",
                       "Merge key collision on customer_id during SCD2 merge", "scd2_merge"),
                  )),
    _PipelineSpec("demo.events_compaction", "Event stream compaction", "Compacts small event files per partition.",
                  "streaming", ("list_partitions", "compact_files", "update_manifest"), 6, 900_000,
                  (2_000_000, 3_500_000), 0.06, (
                      (FailureCategory.RESOURCE_RUNTIME, "MemoryError", "Out of memory while compacting partition",
                       "compact_files"),
                      (FailureCategory.RESOURCE_RUNTIME, "TimeoutError", "Object store listing timed out after 300 s",
                       "list_partitions"),
                  )),
    _PipelineSpec("demo.finance_ledger_reconcile", "Finance ledger reconciliation", "Reconciles ledger vs payments.",
                  "finance-data", ("extract_ledger", "reconcile", "publish_report"), 24, 300_000, (90_000, 120_000),
                  0.10, (
                      (FailureCategory.DATA_QUALITY, "datasentinel.exceptions.QualityGateError",
                       "Quality gate failed: score 58.7 below threshold 60", "reconcile"),
                      (FailureCategory.UNKNOWN, "RuntimeError", "Unexpected response from upstream service",
                       "extract_ledger"),
                  )),
    _PipelineSpec("demo.ml_feature_backfill", "ML feature backfill", "Recomputes features for the training set.",
                  "ml-platform", ("load_events", "compute_features", "write_feature_store"), 48, 1_800_000,
                  (5_000_000, 8_000_000), 0.12, (
                      (FailureCategory.TRANSFORMATION, "pyarrow.lib.ArrowInvalid",
                       "Could not convert 'n/a' with type str: tried to convert to double", "compute_features"),
                      (FailureCategory.RESOURCE_RUNTIME, "OSError", "No space left on device", "write_feature_store"),
                  )),
)


@dataclass
class _Run:
    summary: RunSummary
    root_id: str
    retry_of: str | None
    requested_by: str | None
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    tasks: list[TaskRunView] = field(default_factory=list)
    failures: list[FailureView] = field(default_factory=list)
    events: list[RunEventView] = field(default_factory=list)
    metrics: RunMetricsView | None = None
    heartbeat_at: datetime | None = None


def _run_id(seed: int, *parts: Any) -> str:
    return uuid.uuid5(_NAMESPACE, ":".join(str(p) for p in (seed, *parts))).hex


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


class DemoBackend:
    """Synthetic observability backend implementing both UI protocols (in memory only)."""

    kind = "demo"
    label = "Synthetic demo"
    is_connected = True

    def __init__(self, seed: int = 42, anchor: datetime | None = None, days: int = 30,
                 stale_after_seconds: int = 900) -> None:
        anchor = anchor or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if anchor.tzinfo is None:
            raise ValueError("anchor must be timezone-aware (UTC)")
        self.seed = seed
        self.anchor = anchor.astimezone(timezone.utc)
        self.now = self.anchor
        self.stale_after = timedelta(seconds=stale_after_seconds)
        self._runs: dict[str, _Run] = {}
        self._pipelines = {s.pipeline_id: s for s in _SPECS}
        self._generate(days)

    # ------------------------------------------------------------------ generation

    def _generate(self, days: int) -> None:
        start = self.anchor - timedelta(days=days)
        for index, spec in enumerate(_SPECS):
            rng = random.Random(self.seed * 1000 + index)
            slot_time = start + timedelta(minutes=rng.randint(0, 59))
            slot = 0
            while slot_time < self.anchor - timedelta(hours=1):
                self._generate_chain(spec, rng, slot, slot_time)
                slot += 1
                slot_time += timedelta(hours=spec.cadence_hours)
        self._add_live_examples()

    def _generate_chain(self, spec: _PipelineSpec, rng: random.Random, slot: int, created: datetime) -> None:
        root_id = _run_id(self.seed, spec.pipeline_id, slot, 1)
        attempt, previous = 1, None
        while True:
            run_id = root_id if attempt == 1 else _run_id(self.seed, spec.pipeline_id, slot, attempt)
            fails = rng.random() < (spec.failure_rate if attempt == 1 else 0.3)
            failure = rng.choice(spec.failures) if fails else None
            trigger = "schedule" if attempt == 1 else "retry"
            self._add_terminal_run(spec, rng, run_id, root_id, previous, attempt, created, trigger, failure)
            if failure is None or attempt >= 3 or rng.random() > 0.6:
                return
            previous, attempt = run_id, attempt + 1
            created = self._runs[run_id].summary.ended_at + timedelta(minutes=rng.randint(10, 45))
            self._runs[previous].events.append(RunEventView(created, "retry_requested", None, None, DEMO_ACTOR,
                                                            {"new_attempt": attempt}))

    def _add_terminal_run(self, spec, rng, run_id, root_id, retry_of, attempt, created, trigger, failure) -> None:
        started = created + timedelta(seconds=rng.randint(2, 40))
        rows_in = rng.randint(*spec.rows)
        cursor, tasks, failed_task = started, [], failure[3] if failure else None
        retries = 0
        for task in spec.tasks:
            task_ms = int(spec.base_duration_ms / len(spec.tasks) * rng.lognormvariate(0, 0.25))
            if failure is None and rng.random() < 0.06:  # a task that succeeded on its second attempt
                end = cursor + timedelta(milliseconds=task_ms // 2)
                tasks.append(TaskRunView(task, 1, TaskStatus.FAILED, cursor, end, task_ms // 2, rows_in, None))
                cursor, retries = end, retries + 1
                tasks.append(TaskRunView(task, 2, TaskStatus.SUCCESS, cursor,
                                         cursor + timedelta(milliseconds=task_ms), task_ms, rows_in, rows_in))
                cursor += timedelta(milliseconds=task_ms)
                continue
            end = cursor + timedelta(milliseconds=task_ms)
            status = TaskStatus.FAILED if task == failed_task else TaskStatus.SUCCESS
            tasks.append(TaskRunView(task, 1, status, cursor, end, task_ms, rows_in,
                                     None if status == TaskStatus.FAILED else rows_in))
            cursor = end
            if status == TaskStatus.FAILED:
                break
        duration_ms = int((cursor - started).total_seconds() * 1000)
        status = RunStatus.FAILED if failure else RunStatus.SUCCESS
        rows_out = None if failure else rows_in - rng.randint(0, max(1, rows_in // 500))
        source = "monotonic" if rng.random() < 0.85 else "wall_clock"
        summary = RunSummary(run_id, spec.pipeline_id, spec.name, status, False, trigger, attempt, created, started,
                             cursor, duration_ms, source, rows_in, rows_out, failure[0] if failure else None, True)
        run = _Run(summary, root_id, retry_of, DEMO_ACTOR if trigger == "retry" else None,
                   parameters={"run_date": created.date().isoformat(), "target_schema": "analytics",
                               "connection": "[REDACTED]"},
                   metadata={"code_version": f"v1.{rng.randint(2, 9)}.{rng.randint(0, 20)}", "synthetic": True},
                   tasks=tasks, heartbeat_at=cursor)
        run.events = [RunEventView(created, "created", None, "PENDING", "scheduler" if trigger != "retry" else DEMO_ACTOR),
                      RunEventView(started, "status_change", "PENDING", "RUNNING", "demo.worker"),
                      RunEventView(cursor, "status_change", "RUNNING", status.value, "demo.worker")]
        if failure:
            category, error_type, message, task = failure
            run.failures = [FailureView(True, category, "classifier", f"v1:{category.value.lower()}", error_type,
                                        message, task, task, cursor,
                                        stack=(f"pipelines.{spec.pipeline_id.split('.')[1]}:{task}:{rng.randint(20, 240)}",
                                               "datasentinel.observability.tracker:track_run:118"))]
        failed_final = {t.task_name for t in tasks if t.status == TaskStatus.FAILED} - {
            t.task_name for t in tasks if t.status == TaskStatus.SUCCESS}
        run.metrics = RunMetricsView(
            duration_ms, source, rows_in, rows_out, "measured",
            rows_in / (duration_ms / 1000) if duration_ms > 0 else None, len(tasks),
            sum(1 for t in tasks if t.status == TaskStatus.FAILED), len(failed_final), retries)
        self._runs[run_id] = run

    def _add_live_examples(self) -> None:
        """A healthy RUNNING run, a stale RUNNING run and a PENDING retry request."""
        for spec_id, minutes_ago, heartbeat_ago, key in (("demo.events_compaction", 12, 1, "live-running"),
                                                         ("demo.ml_feature_backfill", 190, 125, "live-stale")):
            spec = self._pipelines[spec_id]
            run_id = _run_id(self.seed, spec_id, key, 1)
            started = self.anchor - timedelta(minutes=minutes_ago)
            done = spec.tasks[0]
            task = TaskRunView(done, 1, TaskStatus.SUCCESS, started, started + timedelta(minutes=4), 240_000,
                               2_500_000, 2_500_000)
            current = TaskRunView(spec.tasks[1], 1, TaskStatus.RUNNING, started + timedelta(minutes=4), None, None,
                                  2_500_000, None)
            summary = RunSummary(run_id, spec_id, spec.name, RunStatus.RUNNING, False, "schedule", 1,
                                 started - timedelta(seconds=20), started, None, None, "unavailable", 2_500_000, None,
                                 None, True)
            self._runs[run_id] = _Run(summary, run_id, None, None, {"run_date": started.date().isoformat()},
                                      {"synthetic": True}, tasks=[task, current],
                                      events=[RunEventView(summary.created_at, "created", None, "PENDING", "scheduler"),
                                              RunEventView(started, "status_change", "PENDING", "RUNNING", "demo.worker")],
                                      heartbeat_at=self.anchor - timedelta(minutes=heartbeat_ago))
        # A deterministic FAILED execution followed by a PENDING retry request (independent of random outcomes).
        spec = self._pipelines["demo.finance_ledger_reconcile"]
        rng = random.Random(self.seed * 1000 + 99)
        failed_id = _run_id(self.seed, spec.pipeline_id, "live-failed", 1)
        self._add_terminal_run(spec, rng, failed_id, failed_id, None, 1, self.anchor - timedelta(hours=3), "schedule",
                               spec.failures[0])
        self._create_retry(self._runs[failed_id], DEMO_ACTOR, "Upstream ledger export re-delivered",
                           self.anchor - timedelta(minutes=30))
        for run in self._runs.values():
            run.summary = replace(run.summary, is_stale=self._stale(run))

    # ------------------------------------------------------------------ helpers

    def _stale(self, run: _Run) -> bool:
        if run.summary.status != RunStatus.RUNNING:
            return False
        last = run.heartbeat_at or run.summary.started_at
        return last is not None and self.now - last > self.stale_after

    def _chain(self, root_id: str) -> list[_Run]:
        return sorted((r for r in self._runs.values() if r.root_id == root_id), key=lambda r: r.summary.attempt_number)

    def _is_latest(self, run: _Run) -> bool:
        return self._chain(run.root_id)[-1].summary.pipeline_run_id == run.summary.pipeline_run_id

    def _allowed(self, run: _Run) -> AllowedActions:
        status = run.summary.status
        if not self._is_latest(run):
            retry_reason = "Only the latest attempt can be retried."
        elif status not in (RunStatus.FAILED, RunStatus.CANCELLED):
            retry_reason = f"Retries can be requested only for FAILED or CANCELLED runs (this run is {status.value})."
        else:
            retry_reason = None
        cancel_reason = None if status == RunStatus.PENDING else \
            f"Only PENDING runs can be cancelled from the dashboard (this run is {status.value})."
        return AllowedActions(retry_reason is None, retry_reason, cancel_reason is None, cancel_reason)

    def _matches(self, run: _Run, f: RunFilters) -> bool:
        s = run.summary
        if not f.include_synthetic and s.is_synthetic:
            return False
        if not f.include_real and not s.is_synthetic:
            return False
        if f.pipeline_ids and s.pipeline_id not in f.pipeline_ids:
            return False
        if f.statuses and s.status not in f.statuses:
            return False
        if f.trigger_types and s.trigger_type not in f.trigger_types:
            return False
        if f.start and s.created_at < f.start:
            return False
        if f.end and s.created_at >= f.end:
            return False
        return not (f.stale_only and not s.is_stale)

    def _filtered(self, f: RunFilters) -> list[_Run]:
        return [r for r in self._runs.values() if self._matches(r, f)]

    # ------------------------------------------------------------------ ObservabilityDataSource

    def list_pipelines(self) -> list[PipelineInfo]:
        return [PipelineInfo(s.pipeline_id, s.name, s.description, s.owner, False, True) for s in _SPECS]

    def trigger_types(self) -> list[str]:
        return sorted({r.summary.trigger_type for r in self._runs.values()})

    def summary(self, filters: RunFilters) -> ObservabilitySummary:
        runs = [r.summary for r in self._filtered(filters)]
        count = lambda status: sum(1 for s in runs if s.status == status)  # noqa: E731
        terminal = [s for s in runs if s.status in TERMINAL_STATUSES]
        durations = [s.duration_ms for s in terminal if s.duration_ms is not None]
        rows = [s.input_rows for s in runs if s.input_rows is not None]
        return ObservabilitySummary(
            total_runs=len(runs), success_count=count(RunStatus.SUCCESS), failed_count=count(RunStatus.FAILED),
            cancelled_count=count(RunStatus.CANCELLED), running_count=count(RunStatus.RUNNING),
            pending_count=count(RunStatus.PENDING), stale_count=sum(1 for s in runs if s.is_stale),
            success_rate_pct=(count(RunStatus.SUCCESS) / len(terminal) * 100) if terminal else None,
            avg_duration_ms=_mean(durations), p95_duration_ms=_p95(durations),
            total_input_rows=sum(rows) if rows else None, computed_at=self.now)

    def status_counts(self, filters: RunFilters) -> list[StatusCount]:
        counts: dict[tuple[str, RunStatus], int] = {}
        for run in self._filtered(filters):
            key = (run.summary.pipeline_name, run.summary.status)
            counts[key] = counts.get(key, 0) + 1
        return [StatusCount(name, status, n) for (name, status), n in sorted(counts.items(), key=lambda kv: kv[0][0])]

    def pipeline_health(self, filters: RunFilters) -> list[PipelineHealth]:
        health = []
        for spec in _SPECS:
            runs = sorted((r.summary for r in self._filtered(filters) if r.summary.pipeline_id == spec.pipeline_id),
                          key=lambda s: s.created_at)
            if not runs:
                continue
            terminal = [s for s in runs if s.status in TERMINAL_STATUSES]
            failures = [s for s in runs if s.failure_category is not None]
            durations = [s.duration_ms for s in terminal if s.duration_ms is not None]
            health.append(PipelineHealth(
                spec.pipeline_id, spec.name, runs[-1].status, runs[-1].created_at, len(runs),
                sum(1 for s in runs if s.status == RunStatus.FAILED),
                (sum(1 for s in terminal if s.status == RunStatus.SUCCESS) / len(terminal) * 100) if terminal else None,
                _p95(durations), failures[-1].failure_category if failures else None, True))
        return health

    def trends(self, filters: RunFilters) -> list[TrendPoint]:
        buckets: dict[datetime, list[RunSummary]] = {}
        for run in self._filtered(filters):
            day = run.summary.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
            buckets.setdefault(day, []).append(run.summary)
        points = []
        for day in sorted(buckets):
            runs = buckets[day]
            durations = [s.duration_ms for s in runs if s.status in TERMINAL_STATUSES and s.duration_ms]
            throughput = [s.input_rows / (s.duration_ms / 1000) for s in runs
                          if s.status == RunStatus.SUCCESS and s.input_rows and s.duration_ms]
            points.append(TrendPoint(
                day, sum(1 for s in runs if s.status == RunStatus.SUCCESS),
                sum(1 for s in runs if s.status == RunStatus.FAILED),
                sum(1 for s in runs if s.status == RunStatus.CANCELLED),
                sum(1 for s in runs if s.status in (RunStatus.RUNNING, RunStatus.PENDING)),
                _mean(durations), _p95(durations), _mean(throughput)))
        return points

    def list_runs(self, filters: RunFilters, limit: int = 50, offset: int = 0) -> RunPage:
        limit = max(1, min(int(limit), 500))
        runs = sorted((r.summary for r in self._filtered(filters)), key=lambda s: (s.created_at, s.pipeline_run_id),
                      reverse=True)
        return RunPage(tuple(runs[offset:offset + limit]), len(runs))

    def get_run(self, pipeline_run_id: str) -> RunDetail | None:
        run = self._runs.get(pipeline_run_id)
        if run is None:
            return None
        attempts = tuple(AttemptView(r.summary.pipeline_run_id, r.summary.attempt_number, r.summary.status,
                                     r.summary.trigger_type, r.summary.created_at, r.summary.failure_category)
                         for r in self._chain(run.root_id))
        return RunDetail(run.summary, run.root_id, run.requested_by, dict(run.parameters), dict(run.metadata),
                         tuple(run.tasks), tuple(run.failures), tuple(run.events), run.metrics, attempts,
                         self._allowed(run), run.heartbeat_at)

    def failure_distribution(self, filters: RunFilters) -> list[FailureStat]:
        counts: dict[FailureCategory, int] = {}
        for run in self._filtered(filters):
            if run.summary.failure_category:
                counts[run.summary.failure_category] = counts.get(run.summary.failure_category, 0) + 1
        return [FailureStat(c, n) for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]

    def top_failure_types(self, filters: RunFilters, limit: int = 5) -> list[FailureTypeStat]:
        stats: dict[str, list] = {}
        for run in self._filtered(filters):
            for failure in run.failures:
                if failure.is_primary:
                    entry = stats.setdefault(failure.error_type, [failure.category, 0, failure.occurred_at])
                    entry[1] += 1
                    entry[2] = max(entry[2], failure.occurred_at)
        ranked = sorted(stats.items(), key=lambda kv: (-kv[1][1], kv[0]))[:limit]
        return [FailureTypeStat(name, cat, n, last) for name, (cat, n, last) in ranked]

    def last_refreshed_at(self) -> datetime | None:
        return self.now

    # ------------------------------------------------------------------ ObservabilityCommands

    def _create_retry(self, previous: _Run, requested_by: str, reason: str, at: datetime) -> str:
        attempt = previous.summary.attempt_number + 1
        run_id = _run_id(self.seed, previous.summary.pipeline_id, previous.root_id, attempt, "requested")
        s = previous.summary
        summary = RunSummary(run_id, s.pipeline_id, s.pipeline_name, RunStatus.PENDING, False, "retry", attempt, at,
                             None, None, None, "unavailable", None, None, None, True)
        self._runs[run_id] = _Run(summary, previous.root_id, s.pipeline_run_id, requested_by, dict(previous.parameters),
                                  {"synthetic": True, "retry_reason": reason},
                                  events=[RunEventView(at, "created", None, "PENDING", requested_by,
                                                       {"retry_of": s.pipeline_run_id, "reason": reason})])
        previous.events.append(RunEventView(at, "retry_requested", None, None, requested_by,
                                            {"new_pipeline_run_id": run_id, "reason": reason}))
        return run_id

    def request_retry(self, pipeline_run_id: str, requested_by: str, reason: str) -> ActionResult:
        run = self._runs.get(pipeline_run_id)
        if run is None:
            return ActionResult(False, "Pipeline run not found.")
        if not requested_by.strip() or not reason.strip():
            return ActionResult(False, "A requester name and a reason are required.")
        allowed = self._allowed(run)
        if not allowed.can_request_retry:
            return ActionResult(False, allowed.retry_blocked_reason or "Retry not allowed.")
        self.now += timedelta(seconds=1)
        new_id = self._create_retry(run, requested_by.strip()[:100], reason.strip()[:500], self.now)
        return ActionResult(True, f"Retry requested as attempt {run.summary.attempt_number + 1} (PENDING). "
                                  f"{DEMO_NOTICE}", new_id)

    def cancel_pending(self, pipeline_run_id: str, actor: str, reason: str) -> ActionResult:
        run = self._runs.get(pipeline_run_id)
        if run is None:
            return ActionResult(False, "Pipeline run not found.")
        if not actor.strip() or not reason.strip():
            return ActionResult(False, "An actor name and a reason are required.")
        allowed = self._allowed(run)
        if not allowed.can_cancel:
            return ActionResult(False, allowed.cancel_blocked_reason or "Cancellation not allowed.")
        self.now += timedelta(seconds=1)
        run.summary = replace(run.summary, status=RunStatus.CANCELLED, ended_at=self.now)
        run.events.append(RunEventView(self.now, "status_change", "PENDING", "CANCELLED", actor.strip()[:100],
                                       {"reason": reason.strip()[:500]}))
        return ActionResult(True, f"Pending run cancelled. {DEMO_NOTICE}")
