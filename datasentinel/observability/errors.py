"""Observability exception hierarchy.

Every error raised by the observability package derives from :class:`ObservabilityError`, which is
itself a :class:`~datasentinel.exceptions.DataSentinelError`, so callers can handle observability
failures uniformly with the rest of the engine. Messages never contain persisted user values.
"""

from __future__ import annotations

from datasentinel.exceptions import DataSentinelError


class ObservabilityError(DataSentinelError):
    """Base class for all pipeline observability errors."""


class ObservabilityValidationError(ObservabilityError):
    """An input value (identifier, size, enum, unsanitised payload) was rejected before any write."""


class PipelineRunNotFoundError(ObservabilityError):
    """The referenced pipeline run (or task run) does not exist."""


class InvalidTransitionError(ObservabilityError):
    """A lifecycle transition is not allowed by the state machine."""


class TerminalStateError(InvalidTransitionError):
    """The pipeline run or task run is terminal and can no longer change."""


class ActiveTasksError(InvalidTransitionError):
    """A pipeline run cannot complete while one or more of its tasks are still RUNNING."""


class PrimaryFailureError(ObservabilityError):
    """Primary-failure rules were violated (FAILED needs exactly one; SUCCESS/CANCELLED allow none)."""


class RetryNotAllowedError(ObservabilityError):
    """A retry was requested for an attempt that is not the latest FAILED or CANCELLED attempt."""


class IntegrityViolationError(ObservabilityError):
    """The database rejected a write because it would break a stored invariant."""


class ObservabilityDatabaseError(ObservabilityError):
    """The observability database could not be opened or used safely."""


class UnsafeDatabasePathError(ObservabilityDatabaseError):
    """The database path is unsafe (symlink, not a regular file, or not owned by the current user)."""


class MigrationError(ObservabilityDatabaseError):
    """Schema migrations could not be verified or applied (for example, a checksum mismatch)."""


class SchemaVersionError(MigrationError):
    """The database schema is newer than this code supports; it is never silently downgraded."""
