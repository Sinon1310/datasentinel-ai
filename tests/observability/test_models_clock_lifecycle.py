"""Models, clocks and the lifecycle state machines."""

from __future__ import annotations

import itertools

import pytest

from datasentinel.exceptions import DataSentinelError
from datasentinel.observability import errors, lifecycle
from datasentinel.observability.clock import Clock, FixedClock, SystemClock
from datasentinel.observability.models import (
    CategorySource, FailureCategory, FailureRecord, PipelineRunStatus, PipelineSpec, Sanitized, TaskStatus,
    validate_pipeline_id, validate_pipeline_run_id, validate_task_name,
)
from tests.observability.conftest import S

RUN = list(PipelineRunStatus)
TASK = list(TaskStatus)
ALLOWED_RUN = {("PENDING", "RUNNING"), ("PENDING", "FAILED"), ("PENDING", "CANCELLED"),
               ("RUNNING", "SUCCESS"), ("RUNNING", "FAILED"), ("RUNNING", "CANCELLED")}


# ------------------------------------------------------------------------------------------- errors

def test_every_observability_error_is_a_datasentinel_error() -> None:
    classes = [getattr(errors, n) for n in dir(errors) if n.endswith("Error") and n != "DataSentinelError"]
    assert classes and all(issubclass(c, errors.ObservabilityError) and issubclass(c, DataSentinelError)
                           for c in classes)
    assert issubclass(errors.TerminalStateError, errors.InvalidTransitionError)
    assert issubclass(errors.SchemaVersionError, errors.MigrationError)


# ---------------------------------------------------------------------------------------- Sanitized

def test_sanitized_cannot_be_constructed_directly() -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        Sanitized("gsk_live_secret_value")
    with pytest.raises(errors.ObservabilityValidationError):
        Sanitized("x", _seal=object())


def test_sanitized_is_immutable_detached_and_never_echoes_content() -> None:
    payload = {"k": [1, 2]}
    value = S(payload)
    payload["k"].append(3)
    assert value.value == {"k": [1, 2]}
    copy = value.value
    copy["k"].append(9)
    assert value.value == {"k": [1, 2]}
    with pytest.raises(AttributeError):
        value._value = "x"  # type: ignore[misc]
    assert "secret" not in repr(S("secret"))
    assert S({"b": 1, "a": 2}).to_json() == '{"a":2,"b":1}'
    assert S("a") == S("a") and hash(S("a")) == hash(S("a"))


@pytest.mark.parametrize("bad", [object(), float("nan"), {1, 2}])
def test_sanitized_rejects_non_json_content(bad) -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        S(bad)


# --------------------------------------------------------------------------------------- validators

@pytest.mark.parametrize("value", ["orders", "orders.daily", "a1_b-c.d", "x" * 64])
def test_valid_pipeline_ids(value: str) -> None:
    assert validate_pipeline_id(value) == value


@pytest.mark.parametrize("value", ["", "Orders", "-x", "x" * 65, "a b", "a/b", "a;drop", "ä", None, 1])
def test_invalid_pipeline_ids(value) -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        validate_pipeline_id(value)


@pytest.mark.parametrize("value", ["", " lead", "x" * 129, "bad\nname", "drop'); --", "a\x00b"])
def test_invalid_task_names(value: str) -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        validate_task_name(value)


@pytest.mark.parametrize("value", ["A" * 32, "0" * 31, "g" * 32, "../" + "0" * 29])
def test_invalid_pipeline_run_ids(value: str) -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        validate_pipeline_run_id(value)


def test_pipeline_spec_and_failure_record_validate_inputs() -> None:
    with pytest.raises(errors.ObservabilityValidationError):
        PipelineSpec("ok", "bad\x07name")
    with pytest.raises(errors.ObservabilityValidationError):
        PipelineSpec("ok", "Name", tags=("Not A Slug",))
    base = dict(category=FailureCategory.SCHEMA, category_source=CategorySource.CLASSIFIER, classifier_rule="r1",
                classifier_version="v1", error_type="pkg.Error", message=S("m"), is_primary=True)
    FailureRecord(**base)
    for override in ({"error_type": "x y"}, {"classifier_rule": "has space"}, {"stage": "\n"},
                     {"pipeline_task_run_id": "nope"}, {"category": "SCHEMA"}, {"is_primary": 1}):
        with pytest.raises(errors.ObservabilityValidationError):
            FailureRecord(**{**base, **override})


# -------------------------------------------------------------------------------------------- clock

def test_system_clock_uses_utc_epoch_ms_and_monotonic_ns() -> None:
    import time
    clock = SystemClock()
    assert isinstance(clock, Clock)
    assert abs(clock.utc_now_ms() - time.time() * 1000) < 5_000
    a = clock.monotonic_ns()
    assert clock.monotonic_ns() >= a


def test_fixed_clock_advances_both_and_can_skew_wall_only() -> None:
    clock = FixedClock(start_ms=1_000, monotonic_start_ns=0)
    clock.advance(seconds=1.5)
    assert (clock.utc_now_ms(), clock.monotonic_ns()) == (2_500, 1_500_000_000)
    clock.set_wall_ms(100)
    assert (clock.utc_now_ms(), clock.monotonic_ns()) == (100, 1_500_000_000)
    with pytest.raises(ValueError):
        clock.advance(ms=-1)
    with pytest.raises(ValueError):
        FixedClock(start_ms=-1)


# ---------------------------------------------------------------------------------------- lifecycle

@pytest.mark.parametrize("current,target", list(itertools.product(RUN, RUN)))
def test_run_transition_table_is_exactly_the_approved_one(current, target) -> None:
    allowed = (current.value, target.value) in ALLOWED_RUN
    assert lifecycle.can_transition_run(current, target) is allowed
    if allowed:
        lifecycle.validate_run_transition(current, target)
    elif current in lifecycle.TERMINAL_RUN_STATUSES:
        with pytest.raises(errors.TerminalStateError):
            lifecycle.validate_run_transition(current, target)
    else:
        with pytest.raises(errors.InvalidTransitionError):
            lifecycle.validate_run_transition(current, target)


@pytest.mark.parametrize("current,target", list(itertools.product(TASK, TASK)))
def test_task_transitions_only_leave_running(current, target) -> None:
    allowed = current is TaskStatus.RUNNING and target is not TaskStatus.RUNNING
    assert lifecycle.can_transition_task(current, target) is allowed
    if not allowed:
        with pytest.raises(errors.InvalidTransitionError):
            lifecycle.validate_task_transition(current, target)


def test_terminal_and_retryable_sets() -> None:
    assert {s.value for s in lifecycle.TERMINAL_RUN_STATUSES} == {"SUCCESS", "FAILED", "CANCELLED"}
    assert {s.value for s in lifecycle.RETRYABLE_RUN_STATUSES} == {"FAILED", "CANCELLED"}
    assert all(lifecycle.is_terminal_task(s) for s in TASK if s is not TaskStatus.RUNNING)
    assert not lifecycle.is_terminal_run(PipelineRunStatus.RUNNING)


@pytest.mark.parametrize("age_ms,expected", [(899_999, False), (900_000, False), (900_001, True)])
def test_stale_is_derived_at_the_threshold(age_ms: int, expected: bool) -> None:
    now = 10_000_000
    assert lifecycle.is_stale(PipelineRunStatus.RUNNING, heartbeat_at_ms=now - age_ms, started_at_ms=0,
                              now_ms=now, stale_after_ms=900_000) is expected


def test_stale_falls_back_to_start_time_and_only_applies_to_running() -> None:
    kwargs = dict(heartbeat_at_ms=None, started_at_ms=0, now_ms=10**9, stale_after_ms=1)
    assert lifecycle.is_stale(PipelineRunStatus.RUNNING, **kwargs)
    for status in (PipelineRunStatus.PENDING, PipelineRunStatus.SUCCESS, PipelineRunStatus.FAILED):
        assert not lifecycle.is_stale(status, **kwargs)
    with pytest.raises(ValueError):
        lifecycle.is_stale(PipelineRunStatus.RUNNING, heartbeat_at_ms=1, started_at_ms=1, now_ms=2, stale_after_ms=0)
