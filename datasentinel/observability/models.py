"""Typed records and enumerations for pipeline observability.

Naming: "run" already means an AI investigation elsewhere in DataSentinel, so every observability
identifier carries a ``pipeline_`` prefix (``PipelineRun``, ``pipeline_run_id``).

All timestamps are integer UTC epoch milliseconds. Records returned by the store are frozen
dataclasses; JSON payloads are returned as parsed values and were redacted before persistence.

Redaction guarantee: free-form values (actors, messages, parameters, metadata, failure context)
can reach the store only wrapped in :class:`Sanitized`. Instances are created exclusively through
the package-internal sealing function used by the redactor, so writing an unredacted value is
rejected at the API boundary rather than relying on caller discipline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Tuple, TypeVar

from datasentinel.observability.errors import ObservabilityValidationError

# --------------------------------------------------------------------------------------------------
# Enumerations (values are the exact strings stored in the database)
# --------------------------------------------------------------------------------------------------


class PipelineRunStatus(str, Enum):
    """Persisted status of a pipeline run attempt. STALE is never stored (it is derived for display)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskStatus(str, Enum):
    """Persisted status of one task attempt inside a pipeline run."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class TriggerType(str, Enum):
    """How a pipeline run attempt came into existence."""

    MANUAL = "manual"
    SCHEDULE = "schedule"
    API = "api"
    RETRY = "retry"
    INTERNAL = "internal"
    SYNTHETIC = "synthetic"


class FailureCategory(str, Enum):
    """Deterministic failure categories."""

    SCHEMA = "SCHEMA"
    DATA_QUALITY = "DATA_QUALITY"
    INGESTION = "INGESTION"
    TRANSFORMATION = "TRANSFORMATION"
    RESOURCE_RUNTIME = "RESOURCE_RUNTIME"
    UNKNOWN = "UNKNOWN"


class CategorySource(str, Enum):
    """Whether a failure category was classified by DataSentinel or reported by the pipeline."""

    CLASSIFIER = "classifier"
    REPORTED = "reported"


class RowsSource(str, Enum):
    """Whether row counts were measured by DataSentinel or reported by the pipeline."""

    MEASURED = "measured"
    REPORTED = "reported"


class DurationSource(str, Enum):
    """Which clock a derived duration came from."""

    MONOTONIC = "monotonic"
    WALL_CLOCK = "wall_clock"
    UNAVAILABLE = "unavailable"


class EventType(str, Enum):
    """Append-only event log entry types."""

    CREATED = "created"
    STATUS_CHANGE = "status_change"
    RETRY_REQUESTED = "retry_requested"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    ROWS_RECORDED = "rows_recorded"
    NOTE = "note"


# --------------------------------------------------------------------------------------------------
# Sanitized wrapper
# --------------------------------------------------------------------------------------------------

T = TypeVar("T")
_SEAL = object()  # capability token: only code holding it can construct Sanitized values


class Sanitized(Generic[T]):
    """A value that has passed the write-path redactor.

    It is immutable (its JSON-compatible content is copied on creation) and cannot be constructed
    directly: ``Sanitized("secret")`` raises. The store accepts free-form content only in this form.
    """

    __slots__ = ("_value",)

    def __init__(self, value: T, *, _seal: object = None) -> None:
        if _seal is not _SEAL:
            raise ObservabilityValidationError(
                "Sanitized values can only be produced by the observability redactor")
        try:
            # Round-trip through JSON: rejects non-JSON content and detaches from caller-owned objects.
            frozen = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ObservabilityValidationError("Sanitized values must be JSON-compatible") from exc
        object.__setattr__(self, "_value", frozen)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Sanitized values are immutable")

    @property
    def value(self) -> T:
        """A copy of the sanitised content (mutating it never affects the wrapper)."""
        return json.loads(json.dumps(self._value))  # type: ignore[attr-defined]

    def to_json(self) -> str:
        """Compact, key-sorted JSON encoding used for persistence."""
        return json.dumps(self._value, ensure_ascii=False, sort_keys=True,  # type: ignore[attr-defined]
                          separators=(",", ":"))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sanitized) and other.to_json() == self.to_json()

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __repr__(self) -> str:  # never echo content into logs
        return f"Sanitized(<{type(self._value).__name__}>)"  # type: ignore[attr-defined]


def _seal_sanitized(value: T) -> Sanitized[T]:
    """Package-internal: wrap a value that has ALREADY been redacted.

    Only the write-path redactor (Milestone 3 Phase 2) may call this. It performs no redaction itself.
    """
    return Sanitized(value, _seal=_SEAL)


# --------------------------------------------------------------------------------------------------
# Identifier validation
# --------------------------------------------------------------------------------------------------

PIPELINE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/#-]{0,127}$")
PIPELINE_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,47}$")
CODE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:/-]{0,63}$")
ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,199}$")
RULE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def validate_pipeline_id(value: str) -> str:
    """Pipeline ids are lowercase slugs: ``^[a-z0-9][a-z0-9._-]*$``, 1–64 characters."""
    if not isinstance(value, str) or not PIPELINE_ID_PATTERN.fullmatch(value):
        raise ObservabilityValidationError(
            "pipeline_id must be a lowercase slug of 1-64 characters (letters, digits, '.', '_', '-')")
    return value


def validate_task_name(value: str) -> str:
    """Task names: 1–128 characters of letters, digits, spaces and ``._:/#-``."""
    if not isinstance(value, str) or not TASK_NAME_PATTERN.fullmatch(value):
        raise ObservabilityValidationError(
            "task_name must be 1-128 characters of letters, digits, spaces and '._:/#-'")
    return value


def validate_pipeline_run_id(value: str) -> str:
    """Pipeline run ids are 32 lowercase hex characters (uuid4/uuid5 ``.hex``)."""
    if not isinstance(value, str) or not PIPELINE_RUN_ID_PATTERN.fullmatch(value):
        raise ObservabilityValidationError("pipeline_run_id must be 32 lowercase hexadecimal characters")
    return value


def validate_display_name(value: str) -> str:
    """Display names: 1–200 printable characters."""
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or _CONTROL_CHARS.search(value):
        raise ObservabilityValidationError("display_name must be 1-200 printable characters")
    return value


def validate_tag(value: str) -> str:
    """Tags are lowercase slugs of up to 48 characters."""
    if not isinstance(value, str) or not TAG_PATTERN.fullmatch(value):
        raise ObservabilityValidationError("tags must be lowercase slugs of 1-48 characters")
    return value


def validate_code_version(value: str) -> str:
    """Code versions (e.g. a git SHA or semantic version): 1–64 safe characters."""
    if not isinstance(value, str) or not CODE_VERSION_PATTERN.fullmatch(value):
        raise ObservabilityValidationError("code_version must be 1-64 characters of letters, digits and '._+:/-'")
    return value


def validate_non_negative(name: str, value: Optional[int]) -> Optional[int]:
    """Counts and elapsed times must be non-negative integers (``None`` means unknown)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObservabilityValidationError(f"{name} must be a non-negative integer")
    return value


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineSpec:
    """Definition used to register a pipeline the first time one of its runs is recorded."""

    pipeline_id: str
    display_name: str
    description: Optional[Sanitized[str]] = None
    owner: Optional[Sanitized[str]] = None
    tags: Tuple[str, ...] = ()
    is_internal: bool = False
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        validate_pipeline_id(self.pipeline_id)
        validate_display_name(self.display_name)
        if len(self.tags) > 20:
            raise ObservabilityValidationError("a pipeline can have at most 20 tags")
        for tag in self.tags:
            validate_tag(tag)


@dataclass(frozen=True)
class FailureRecord:
    """One failure to persist with :meth:`ObservabilityStore.fail_run`.

    ``message``, ``exception_chain``, ``stack`` and ``context`` must already be redacted
    (``stack`` holds only ``module``/``function``/``lineno`` frames: no source lines, no locals).
    """

    category: FailureCategory
    category_source: CategorySource
    classifier_rule: str
    classifier_version: str
    error_type: str
    message: Sanitized[str]
    is_primary: bool
    exception_chain: Optional[Sanitized[list]] = None
    stack: Optional[Sanitized[list]] = None
    context: Optional[Sanitized[dict]] = None
    stage: Optional[str] = None
    pipeline_task_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, FailureCategory) or not isinstance(self.category_source, CategorySource):
            raise ObservabilityValidationError("category and category_source must use the observability enums")
        if not isinstance(self.error_type, str) or not ERROR_TYPE_PATTERN.fullmatch(self.error_type):
            raise ObservabilityValidationError("error_type must be a qualified class name (1-200 characters)")
        for name in ("classifier_rule", "classifier_version"):
            if not isinstance(getattr(self, name), str) or not RULE_PATTERN.fullmatch(getattr(self, name)):
                raise ObservabilityValidationError(f"{name} must be 1-64 characters of letters, digits and '._:-'")
        if self.stage is not None:
            validate_task_name(self.stage)
        if self.pipeline_task_run_id is not None:
            validate_pipeline_run_id(self.pipeline_task_run_id)
        if not isinstance(self.is_primary, bool):
            raise ObservabilityValidationError("is_primary must be a bool")


# --------------------------------------------------------------------------------------------------
# Stored records
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline:
    """A registered pipeline."""

    pipeline_id: str
    display_name: str
    description: Optional[str]
    owner: Optional[str]
    tags: Tuple[str, ...]
    is_internal: bool
    is_synthetic: bool
    created_at_ms: int


@dataclass(frozen=True)
class PipelineRun:
    """One attempt of a pipeline execution (retries are separate, linked attempts)."""

    pipeline_run_id: str
    pipeline_id: str
    root_pipeline_run_id: str
    retry_of_pipeline_run_id: Optional[str]
    attempt_number: int
    status: PipelineRunStatus
    status_changed_by: str
    status_changed_at_ms: int
    trigger_type: TriggerType
    requested_by: Optional[str]
    retry_reason: Optional[str]
    created_at_ms: int
    started_at_ms: Optional[int]
    ended_at_ms: Optional[int]
    monotonic_elapsed_ms: Optional[int]
    heartbeat_at_ms: Optional[int]
    input_rows: Optional[int]
    output_rows: Optional[int]
    rows_source: Optional[RowsSource]
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    code_version: Optional[str] = None
    is_synthetic: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.status in (PipelineRunStatus.SUCCESS, PipelineRunStatus.FAILED, PipelineRunStatus.CANCELLED)


@dataclass(frozen=True)
class PipelineTaskRun:
    """One attempt of a task inside a pipeline run."""

    pipeline_task_run_id: str
    pipeline_run_id: str
    task_name: str
    task_attempt: int
    status: TaskStatus
    started_at_ms: int
    ended_at_ms: Optional[int]
    monotonic_elapsed_ms: Optional[int]
    input_rows: Optional[int]
    output_rows: Optional[int]
    rows_source: Optional[RowsSource]


@dataclass(frozen=True)
class PipelineFailure:
    """A persisted (already redacted) failure record."""

    pipeline_failure_id: str
    pipeline_run_id: str
    pipeline_task_run_id: Optional[str]
    is_primary: bool
    category: FailureCategory
    category_source: CategorySource
    classifier_rule: str
    classifier_version: str
    error_type: str
    error_message: str
    exception_chain: list
    stack: list
    stage: Optional[str]
    context: dict
    occurred_at_ms: int


@dataclass(frozen=True)
class PipelineRunEvent:
    """An append-only event-log entry."""

    event_id: int
    pipeline_run_id: str
    at_ms: int
    event_type: EventType
    from_status: Optional[str]
    to_status: Optional[str]
    actor: str
    details: dict


@dataclass(frozen=True)
class PipelineRunMetrics:
    """Derived metrics, written once when the run reaches a terminal state."""

    pipeline_run_id: str
    metrics_version: str
    duration_ms: Optional[int]
    duration_source: DurationSource
    input_rows: Optional[int]
    output_rows: Optional[int]
    rows_source: Optional[RowsSource]
    input_throughput_rows_per_s: Optional[float]
    task_attempt_count: int
    failed_task_attempt_count: int
    failed_task_count: int
    task_retry_count: int
    computed_at_ms: int
