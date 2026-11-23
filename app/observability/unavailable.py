"""The "live" data source until the observability backend exists: empty and honest."""

from __future__ import annotations

from datetime import datetime, timezone

from app.observability.contracts import (
    ActionResult,
    FailureStat,
    FailureTypeStat,
    ObservabilitySummary,
    PipelineHealth,
    PipelineInfo,
    RunDetail,
    RunFilters,
    RunPage,
    StatusCount,
    TrendPoint,
)

NOT_CONNECTED = ("The pipeline observability backend is not connected yet. Run tracking, history and failure "
                 "classification are being built in Milestone 3.")


class UnavailableBackend:
    """Implements both protocols; returns no records and refuses every action."""

    kind = "live"
    label = "Live backend"
    is_connected = False

    def list_pipelines(self) -> list[PipelineInfo]:
        return []

    def trigger_types(self) -> list[str]:
        return []

    def summary(self, filters: RunFilters) -> ObservabilitySummary:
        return ObservabilitySummary(0, 0, 0, 0, 0, 0, 0, None, None, None, None, datetime.now(timezone.utc))

    def status_counts(self, filters: RunFilters) -> list[StatusCount]:
        return []

    def pipeline_health(self, filters: RunFilters) -> list[PipelineHealth]:
        return []

    def trends(self, filters: RunFilters) -> list[TrendPoint]:
        return []

    def list_runs(self, filters: RunFilters, limit: int = 50, offset: int = 0) -> RunPage:
        return RunPage((), 0)

    def get_run(self, pipeline_run_id: str) -> RunDetail | None:
        return None

    def failure_distribution(self, filters: RunFilters) -> list[FailureStat]:
        return []

    def top_failure_types(self, filters: RunFilters, limit: int = 5) -> list[FailureTypeStat]:
        return []

    def last_refreshed_at(self) -> datetime | None:
        return None

    def request_retry(self, pipeline_run_id: str, requested_by: str, reason: str) -> ActionResult:
        return ActionResult(False, NOT_CONNECTED)

    def cancel_pending(self, pipeline_run_id: str, actor: str, reason: str) -> ActionResult:
        return ActionResult(False, NOT_CONNECTED)
