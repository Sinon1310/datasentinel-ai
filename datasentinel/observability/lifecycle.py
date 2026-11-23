"""Explicit lifecycle state machines for pipeline runs and task runs.

These tables are the single code-level definition of allowed transitions. The database enforces the
same rules independently with triggers (see :mod:`datasentinel.observability.migrations`); tests
assert that both layers agree for every (from, to) pair.

Pipeline run::

    PENDING ──▶ RUNNING ──▶ SUCCESS
       │           ├──────▶ FAILED      (requires exactly one primary failure)
       ├──────────▶├──────▶ CANCELLED
       └─▶ FAILED / CANCELLED (before starting)

    SUCCESS, FAILED and CANCELLED are terminal and immutable.

Task run::

    RUNNING ──▶ SUCCESS | FAILED | SKIPPED | CANCELLED   (then final)

STALE is a *derived display state* for RUNNING runs whose heartbeat is too old. It is never
persisted, and deriving it never changes the stored status.
"""

from __future__ import annotations

from typing import FrozenSet, Mapping, Optional

from datasentinel.observability.errors import InvalidTransitionError, TerminalStateError
from datasentinel.observability.models import PipelineRunStatus, TaskStatus

_R = PipelineRunStatus
_T = TaskStatus

RUN_TRANSITIONS: Mapping[PipelineRunStatus, FrozenSet[PipelineRunStatus]] = {
    _R.PENDING: frozenset({_R.RUNNING, _R.FAILED, _R.CANCELLED}),
    _R.RUNNING: frozenset({_R.SUCCESS, _R.FAILED, _R.CANCELLED}),
    _R.SUCCESS: frozenset(),
    _R.FAILED: frozenset(),
    _R.CANCELLED: frozenset(),
}
TERMINAL_RUN_STATUSES: FrozenSet[PipelineRunStatus] = frozenset({_R.SUCCESS, _R.FAILED, _R.CANCELLED})
INITIAL_RUN_STATUSES: FrozenSet[PipelineRunStatus] = frozenset({_R.PENDING, _R.RUNNING})
RETRYABLE_RUN_STATUSES: FrozenSet[PipelineRunStatus] = frozenset({_R.FAILED, _R.CANCELLED})

TASK_TRANSITIONS: Mapping[TaskStatus, FrozenSet[TaskStatus]] = {
    _T.RUNNING: frozenset({_T.SUCCESS, _T.FAILED, _T.SKIPPED, _T.CANCELLED}),
    _T.SUCCESS: frozenset(),
    _T.FAILED: frozenset(),
    _T.SKIPPED: frozenset(),
    _T.CANCELLED: frozenset(),
}
TERMINAL_TASK_STATUSES: FrozenSet[TaskStatus] = frozenset({_T.SUCCESS, _T.FAILED, _T.SKIPPED, _T.CANCELLED})


def is_terminal_run(status: PipelineRunStatus) -> bool:
    """True for SUCCESS, FAILED and CANCELLED."""
    return PipelineRunStatus(status) in TERMINAL_RUN_STATUSES


def is_terminal_task(status: TaskStatus) -> bool:
    """True for every task status except RUNNING."""
    return TaskStatus(status) in TERMINAL_TASK_STATUSES


def can_transition_run(current: PipelineRunStatus, target: PipelineRunStatus) -> bool:
    """Whether ``current → target`` is an allowed pipeline-run transition."""
    return PipelineRunStatus(target) in RUN_TRANSITIONS[PipelineRunStatus(current)]


def can_transition_task(current: TaskStatus, target: TaskStatus) -> bool:
    """Whether ``current → target`` is an allowed task-run transition."""
    return TaskStatus(target) in TASK_TRANSITIONS[TaskStatus(current)]


def validate_run_transition(current: PipelineRunStatus, target: PipelineRunStatus) -> None:
    """Raise unless ``current → target`` is allowed.

    Raises:
        TerminalStateError: ``current`` is terminal (terminal runs never change).
        InvalidTransitionError: the transition is not in :data:`RUN_TRANSITIONS`.
    """
    current, target = PipelineRunStatus(current), PipelineRunStatus(target)
    if current in TERMINAL_RUN_STATUSES:
        raise TerminalStateError(f"Pipeline run is {current.value} (terminal) and cannot change to {target.value}.")
    if target not in RUN_TRANSITIONS[current]:
        raise InvalidTransitionError(f"Pipeline run cannot move from {current.value} to {target.value}.")


def validate_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    """Raise unless ``current → target`` is an allowed task transition."""
    current, target = TaskStatus(current), TaskStatus(target)
    if current in TERMINAL_TASK_STATUSES:
        raise TerminalStateError(f"Task run is {current.value} (terminal) and cannot change to {target.value}.")
    if target not in TASK_TRANSITIONS[current]:
        raise InvalidTransitionError(f"Task run cannot move from {current.value} to {target.value}.")


def is_stale(status: PipelineRunStatus, *, heartbeat_at_ms: Optional[int], started_at_ms: Optional[int],
             now_ms: int, stale_after_ms: int) -> bool:
    """Derived STALE display state: a RUNNING run with no sign of life for longer than the threshold.

    The last sign of life is the latest heartbeat, or the start time if no heartbeat was ever sent.
    A run exactly at the threshold is not stale. This function is pure; it never changes stored data.
    """
    if stale_after_ms <= 0:
        raise ValueError("stale_after_ms must be positive")
    if PipelineRunStatus(status) is not PipelineRunStatus.RUNNING:
        return False
    last_seen = heartbeat_at_ms if heartbeat_at_ms is not None else started_at_ms
    if last_seen is None:
        return False
    return now_ms - last_seen > stale_after_ms
