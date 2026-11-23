"""Pipeline observability engine (Milestone 3).

Phase 1 (foundations): typed models, lifecycle state machines, clocks, versioned migrations and the
transactional SQLite store. Phase 2 (writer and safety): write-path redaction, deterministic failure
classification, metric formulas, quality gates and the PipelineTracker writer facade. The read-only
reader, the synthetic demo and the dashboard integration arrive in later phases.

This package is part of the deterministic engine: it never imports the agent layer, LLM clients or UI.
"""

from datasentinel.observability.clock import Clock, FixedClock, SystemClock
from datasentinel.observability.errors import (
    ActiveTasksError, IntegrityViolationError, InvalidTransitionError, MigrationError, ObservabilityDatabaseError,
    ObservabilityError, ObservabilityValidationError, PipelineRunNotFoundError, PrimaryFailureError,
    RetryNotAllowedError, SchemaVersionError, TerminalStateError, UnsafeDatabasePathError,
)
from datasentinel.observability.models import (
    CategorySource, DurationSource, EventType, FailureCategory, FailureRecord, Pipeline, PipelineFailure,
    PipelineRun, PipelineRunEvent, PipelineRunMetrics, PipelineRunStatus, PipelineSpec, PipelineTaskRun,
    RowsSource, Sanitized, TaskStatus, TriggerType,
)
from datasentinel.observability.classifier import Classification, ClassifierConfig, FailureClassifier
from datasentinel.observability.gates import GateConfig, GateResult, enforce_quality_gate, evaluate_quality_gate
from datasentinel.observability.redaction import Redactor
from datasentinel.observability.store import ObservabilityStore
from datasentinel.observability.tracker import PipelineTracker, RunHandle, TaskHandle

__all__ = [
    "Classification", "ClassifierConfig", "FailureClassifier", "GateConfig", "GateResult", "PipelineTracker",
    "Redactor", "RunHandle", "TaskHandle", "enforce_quality_gate", "evaluate_quality_gate",
    "ActiveTasksError", "CategorySource", "Clock", "DurationSource", "EventType", "FailureCategory",
    "FailureRecord", "FixedClock", "IntegrityViolationError", "InvalidTransitionError", "MigrationError",
    "ObservabilityDatabaseError", "ObservabilityError", "ObservabilityStore", "ObservabilityValidationError",
    "Pipeline", "PipelineFailure", "PipelineRun", "PipelineRunEvent", "PipelineRunMetrics",
    "PipelineRunNotFoundError", "PipelineRunStatus", "PipelineSpec", "PipelineTaskRun", "PrimaryFailureError",
    "RetryNotAllowedError", "RowsSource", "Sanitized", "SchemaVersionError", "SystemClock", "TaskStatus",
    "TerminalStateError", "TriggerType", "UnsafeDatabasePathError",
]
