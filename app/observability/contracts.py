"""Typed view models and protocols for the Pipeline Observability page.

Field names follow the approved Milestone 3 design (``pipeline_run_id``, ``attempt_number``,
``duration_source``, failure categories, task statuses) so the real backend can be adapted later
without changing the page. All aggregates (KPIs, trends, health) are produced by the data source —
the UI never derives authoritative metrics itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable


class RunStatus(str, Enum):
    """Persisted pipeline-run status."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset({RunStatus.SUCCESS, RunStatus.FAILED, RunStatus.CANCELLED})


class TaskStatus(str, Enum):
    """Task-run status."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class FailureCategory(str, Enum):
    """Deterministic failure classification categories."""

    SCHEMA = "SCHEMA"
    DATA_QUALITY = "DATA_QUALITY"
    INGESTION = "INGESTION"
    TRANSFORMATION = "TRANSFORMATION"
    RESOURCE_RUNTIME = "RESOURCE_RUNTIME"
    UNKNOWN = "UNKNOWN"


TriggerType = Literal["manual", "schedule", "api", "retry", "internal", "synthetic"]
DurationSource = Literal["monotonic", "wall_clock", "unavailable"]
DataSourceKind = Literal["live", "demo"]


@dataclass(frozen=True)
class RunFilters:
    """Filters applied by the data source (UTC bounds are inclusive start, exclusive end)."""

    pipeline_ids: tuple[str, ...] = ()
    statuses: tuple[RunStatus, ...] = ()
    trigger_types: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    include_synthetic: bool = True
    include_real: bool = True
    stale_only: bool = False


@dataclass(frozen=True)
class PipelineInfo:
    pipeline_id: str
    display_name: str
    description: str | None = None
    owner: str | None = None
    is_internal: bool = False
    is_synthetic: bool = False


@dataclass(frozen=True)
class RunSummary:
    """One row of execution history. ``is_stale`` is derived (display only); ``status`` is persisted."""

    pipeline_run_id: str
    pipeline_id: str
    pipeline_name: str
    status: RunStatus
    is_stale: bool
    trigger_type: str
    attempt_number: int
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    duration_ms: int | None
    duration_source: DurationSource
    input_rows: int | None
    output_rows: int | None
    failure_category: FailureCategory | None
    is_synthetic: bool

    @property
    def display_status(self) -> str:
        """Status as shown: persisted status, annotated when derived-stale."""
        return f"{self.status.value} · STALE" if self.is_stale else self.status.value


@dataclass(frozen=True)
class RunPage:
    items: tuple[RunSummary, ...]
    total: int


@dataclass(frozen=True)
class ObservabilitySummary:
    """Authoritative aggregates for the current filters (computed by the data source)."""

    total_runs: int
    success_count: int
    failed_count: int
    cancelled_count: int
    running_count: int
    pending_count: int
    stale_count: int
    success_rate_pct: float | None        # SUCCESS ÷ terminal runs; None when no terminal runs
    avg_duration_ms: float | None         # terminal runs with a duration
    p95_duration_ms: float | None
    total_input_rows: int | None          # sum of known input rows; None when none known
    computed_at: datetime


@dataclass(frozen=True)
class PipelineHealth:
    pipeline_id: str
    pipeline_name: str
    last_status: RunStatus | None
    last_run_at: datetime | None
    runs: int
    failed: int
    success_rate_pct: float | None
    p95_duration_ms: float | None
    last_failure_category: FailureCategory | None
    is_synthetic: bool


@dataclass(frozen=True)
class StatusCount:
    pipeline_name: str
    status: RunStatus
    count: int


@dataclass(frozen=True)
class TrendPoint:
    """Daily bucket (UTC)."""

    bucket: datetime
    success: int
    failed: int
    cancelled: int
    active: int
    avg_duration_ms: float | None
    p95_duration_ms: float | None
    avg_throughput_rows_per_s: float | None


@dataclass(frozen=True)
class FailureStat:
    category: FailureCategory
    count: int


@dataclass(frozen=True)
class FailureTypeStat:
    error_type: str
    category: FailureCategory
    count: int
    last_seen: datetime


@dataclass(frozen=True)
class TaskRunView:
    task_name: str
    task_attempt: int
    status: TaskStatus
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int | None
    input_rows: int | None
    output_rows: int | None


@dataclass(frozen=True)
class FailureView:
    """A classified failure; messages are redacted before persistence by the backend."""

    is_primary: bool
    category: FailureCategory
    category_source: Literal["classifier", "reported"]
    classifier_rule: str
    error_type: str
    error_message: str
    stage: str | None
    task_name: str | None
    occurred_at: datetime
    stack: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunEventView:
    at: datetime
    event_type: str
    from_status: str | None
    to_status: str | None
    actor: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunMetricsView:
    duration_ms: int | None
    duration_source: DurationSource
    input_rows: int | None
    output_rows: int | None
    rows_source: str | None
    input_throughput_rows_per_s: float | None
    task_attempt_count: int
    failed_task_attempt_count: int
    failed_task_count: int
    task_retry_count: int


@dataclass(frozen=True)
class AttemptView:
    pipeline_run_id: str
    attempt_number: int
    status: RunStatus
    trigger_type: str
    created_at: datetime
    failure_category: FailureCategory | None


@dataclass(frozen=True)
class AllowedActions:
    """What the backend permits for this run. The UI disables anything not allowed."""

    can_request_retry: bool
    retry_blocked_reason: str | None
    can_cancel: bool
    cancel_blocked_reason: str | None


@dataclass(frozen=True)
class RunDetail:
    run: RunSummary
    root_pipeline_run_id: str
    requested_by: str | None
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    tasks: tuple[TaskRunView, ...]
    failures: tuple[FailureView, ...]
    events: tuple[RunEventView, ...]
    metrics: RunMetricsView | None
    attempts: tuple[AttemptView, ...]
    allowed_actions: AllowedActions
    heartbeat_at: datetime | None


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str
    created_pipeline_run_id: str | None = None


@runtime_checkable
class ObservabilityDataSource(Protocol):
    """Read-only access for display."""

    kind: DataSourceKind
    label: str
    is_connected: bool

    def list_pipelines(self) -> list[PipelineInfo]: ...
    def trigger_types(self) -> list[str]: ...
    def summary(self, filters: RunFilters) -> ObservabilitySummary: ...
    def status_counts(self, filters: RunFilters) -> list[StatusCount]: ...
    def pipeline_health(self, filters: RunFilters) -> list[PipelineHealth]: ...
    def trends(self, filters: RunFilters) -> list[TrendPoint]: ...
    def list_runs(self, filters: RunFilters, limit: int = 50, offset: int = 0) -> RunPage: ...
    def get_run(self, pipeline_run_id: str) -> RunDetail | None: ...
    def failure_distribution(self, filters: RunFilters) -> list[FailureStat]: ...
    def top_failure_types(self, filters: RunFilters, limit: int = 5) -> list[FailureTypeStat]: ...
    def last_refreshed_at(self) -> datetime | None: ...


@runtime_checkable
class ObservabilityCommands(Protocol):
    """The only writes the UI may request. Neither executes a pipeline."""

    def request_retry(self, pipeline_run_id: str, requested_by: str, reason: str) -> ActionResult: ...
    def cancel_pending(self, pipeline_run_id: str, actor: str, reason: str) -> ActionResult: ...
