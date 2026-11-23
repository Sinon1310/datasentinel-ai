"""Store operations: lifecycle transitions, tasks, failures, retries, metrics and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasentinel.observability import (
    ActiveTasksError, DurationSource, EventType, FailureCategory, FixedClock, InvalidTransitionError, ObservabilityStore,
    ObservabilityValidationError, PipelineRunNotFoundError, PipelineRunStatus, PipelineSpec, PrimaryFailureError,
    RetryNotAllowedError, RowsSource, TaskStatus, TerminalStateError, TriggerType,
)
from tests.observability.conftest import ACTOR, SPEC, S, failure

R = PipelineRunStatus


def _status_events(store: ObservabilityStore, run_id: str) -> list:
    return [(e.from_status, e.to_status, e.actor) for e in store.get_events(run_id)
            if e.event_type is EventType.STATUS_CHANGE]


# ------------------------------------------------------------------------------------------ creation

def test_create_pending_run_registers_pipeline_and_logs_created(store, clock) -> None:
    run = store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.SCHEDULE,
                           parameters=S({"date": "2026-11-23"}), metadata=S({"env": "test"}), code_version="abc123")
    assert run.status is R.PENDING and run.attempt_number == 1
    assert run.root_pipeline_run_id == run.pipeline_run_id and run.retry_of_pipeline_run_id is None
    assert run.started_at_ms is None and run.created_at_ms == clock.utc_now_ms()
    assert run.parameters == {"date": "2026-11-23"} and run.metadata == {"env": "test"}
    assert store.get_pipeline("orders.daily").display_name == "Orders daily load"
    [event] = store.get_events(run.pipeline_run_id)
    assert event.event_type is EventType.CREATED and event.details["initial_status"] == "PENDING"


def test_create_running_run_sets_started(store, clock) -> None:
    run = store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.MANUAL, start=True)
    assert run.status is R.RUNNING and run.started_at_ms == clock.utc_now_ms()


def test_create_run_rejects_retry_trigger_and_bad_synthetic_use(store) -> None:
    with pytest.raises(ObservabilityValidationError, match="request_retry"):
        store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.RETRY)
    with pytest.raises(ObservabilityValidationError, match="synthetic"):
        store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.SYNTHETIC)
    store.register_pipeline(SPEC)
    with pytest.raises(ObservabilityValidationError, match="different"):
        store.create_run(PipelineSpec("orders.daily", "Orders", is_synthetic=True), actor=ACTOR,
                         trigger_type=TriggerType.SYNTHETIC)


def test_registration_is_idempotent_and_immutable(store) -> None:
    first = store.register_pipeline(SPEC)
    again = store.register_pipeline(PipelineSpec("orders.daily", "Renamed"))
    assert again == first


def test_create_run_with_explicit_id_and_duplicate_rejected(store) -> None:
    run_id = "a" * 32
    assert store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.API, pipeline_run_id=run_id).pipeline_run_id \
        == run_id
    with pytest.raises(Exception) as info:
        store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.API, pipeline_run_id=run_id)
    assert "UNIQUE" in str(info.value)


def test_unknown_run_raises_not_found(store) -> None:
    with pytest.raises(PipelineRunNotFoundError):
        store.get_run("f" * 32)
    with pytest.raises(PipelineRunNotFoundError):
        store.start_run("f" * 32, actor=ACTOR)


# ------------------------------------------------------------------------------------- transitions

def test_start_run_is_compare_and_set(store, new_run) -> None:
    run = new_run(start=False)
    started = store.start_run(run.pipeline_run_id, actor=S("claimer"))
    assert started.status is R.RUNNING and started.status_changed_by == "claimer"
    with pytest.raises(InvalidTransitionError):
        store.start_run(run.pipeline_run_id, actor=ACTOR)
    assert _status_events(store, run.pipeline_run_id) == [("PENDING", "RUNNING", "claimer")]


def test_complete_run_records_status_event_and_metrics(store, new_run, clock) -> None:
    run = new_run()
    clock.advance(seconds=4)
    done = store.complete_run(run.pipeline_run_id, actor=ACTOR)
    assert done.status is R.SUCCESS and done.ended_at_ms == done.started_at_ms + 4_000
    assert _status_events(store, run.pipeline_run_id) == [("RUNNING", "SUCCESS", "test.runner")]
    metrics = store.get_metrics(run.pipeline_run_id)
    assert (metrics.duration_ms, metrics.duration_source) == (4_000, DurationSource.WALL_CLOCK)


def test_complete_from_pending_is_invalid(store, new_run) -> None:
    run = new_run(start=False)
    with pytest.raises(InvalidTransitionError):
        store.complete_run(run.pipeline_run_id, actor=ACTOR)
    assert store.get_run(run.pipeline_run_id).status is R.PENDING
    assert store.get_metrics(run.pipeline_run_id) is None


def test_complete_is_rejected_while_tasks_are_running(store, new_run) -> None:
    run = new_run()
    task = store.start_task(run.pipeline_run_id, "load", actor=ACTOR)
    with pytest.raises(ActiveTasksError, match="1 task"):
        store.complete_run(run.pipeline_run_id, actor=ACTOR)
    assert store.get_run(run.pipeline_run_id).status is R.RUNNING
    assert store.get_metrics(run.pipeline_run_id) is None
    store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    assert store.complete_run(run.pipeline_run_id, actor=ACTOR).status is R.SUCCESS


@pytest.mark.parametrize("terminal", ["complete", "fail", "cancel"])
def test_terminal_runs_are_immutable_through_every_operation(store, new_run, terminal: str) -> None:
    run = new_run()
    rid = run.pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    {"complete": lambda: store.complete_run(rid, actor=ACTOR),
     "fail": lambda: store.fail_run(rid, [failure()], actor=ACTOR),
     "cancel": lambda: store.cancel_run(rid, actor=ACTOR)}[terminal]()
    before = (store.get_run(rid), store.get_events(rid), store.get_tasks(rid), store.get_metrics(rid))
    attempts = [
        lambda: store.start_run(rid, actor=ACTOR),
        lambda: store.complete_run(rid, actor=ACTOR),
        lambda: store.fail_run(rid, [failure()], actor=ACTOR),
        lambda: store.cancel_run(rid, actor=ACTOR),
        lambda: store.heartbeat(rid),
        lambda: store.record_rows(rid, input_rows=1, output_rows=1, rows_source=RowsSource.REPORTED, actor=ACTOR),
        lambda: store.start_task(rid, "extract2", actor=ACTOR),
        lambda: store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR),
    ]
    for attempt in attempts:
        with pytest.raises(TerminalStateError):
            attempt()
    assert (store.get_run(rid), store.get_events(rid), store.get_tasks(rid), store.get_metrics(rid)) == before


def test_heartbeat_updates_running_runs_only_without_events(store, new_run, clock) -> None:
    pending, running = new_run(start=False), new_run()
    clock.advance(seconds=30)
    store.heartbeat(running.pipeline_run_id)
    assert store.get_run(running.pipeline_run_id).heartbeat_at_ms == clock.utc_now_ms()
    assert len(store.get_events(running.pipeline_run_id)) == 1  # only "created"
    with pytest.raises(InvalidTransitionError):
        store.heartbeat(pending.pipeline_run_id)


def test_record_rows_validates_and_logs(store, new_run) -> None:
    run = new_run()
    with pytest.raises(ObservabilityValidationError):
        store.record_rows(run.pipeline_run_id, input_rows=-1, output_rows=None, rows_source=RowsSource.MEASURED,
                          actor=ACTOR)
    with pytest.raises(ObservabilityValidationError):
        store.record_rows(run.pipeline_run_id, input_rows=None, output_rows=None, rows_source=RowsSource.MEASURED,
                          actor=ACTOR)
    with pytest.raises(ObservabilityValidationError):
        store.record_rows(run.pipeline_run_id, input_rows=True, output_rows=None, rows_source=RowsSource.MEASURED,
                          actor=ACTOR)
    updated = store.record_rows(run.pipeline_run_id, input_rows=10, output_rows=None,
                                rows_source=RowsSource.MEASURED, actor=ACTOR)
    updated = store.record_rows(run.pipeline_run_id, input_rows=None, output_rows=8,
                                rows_source=RowsSource.MEASURED, actor=ACTOR)
    assert (updated.input_rows, updated.output_rows, updated.rows_source) == (10, 8, RowsSource.MEASURED)
    assert store.get_events(run.pipeline_run_id)[-1].event_type is EventType.ROWS_RECORDED


# ------------------------------------------------------------------------------------------- tasks

def test_tasks_need_a_running_parent(store, new_run) -> None:
    pending = new_run(start=False)
    with pytest.raises(InvalidTransitionError):
        store.start_task(pending.pipeline_run_id, "extract", actor=ACTOR)


def test_task_attempts_follow_failure_only(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    first = store.start_task(rid, "extract", actor=ACTOR)
    with pytest.raises(InvalidTransitionError, match="RUNNING"):
        store.start_task(rid, "extract", actor=ACTOR)
    store.finish_task(first.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR)
    second = store.start_task(rid, "extract", actor=ACTOR)
    assert second.task_attempt == 2
    store.finish_task(second.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR,
                      input_rows=5, output_rows=5, rows_source=RowsSource.MEASURED, monotonic_elapsed_ms=12)
    with pytest.raises(InvalidTransitionError, match="SUCCESS"):
        store.start_task(rid, "extract", actor=ACTOR)


def test_finished_tasks_are_immutable(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    with pytest.raises(InvalidTransitionError):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.RUNNING, actor=ACTOR)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.SKIPPED, actor=ACTOR)
    with pytest.raises(TerminalStateError):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    assert store.get_tasks(rid)[0].status is TaskStatus.SKIPPED


def test_task_rows_need_a_source(store, new_run) -> None:
    task = store.start_task(new_run().pipeline_run_id, "extract", actor=ACTOR)
    with pytest.raises(ObservabilityValidationError):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR, input_rows=3)
    with pytest.raises(ObservabilityValidationError):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR,
                          rows_source=RowsSource.MEASURED)
    with pytest.raises(PipelineRunNotFoundError):
        store.finish_task("e" * 32, TaskStatus.SUCCESS, actor=ACTOR)


# ---------------------------------------------------------------------------------------- failures

@pytest.mark.parametrize("records", [[], [failure(False)], [failure(True), failure(True)]])
def test_failed_runs_need_exactly_one_primary_failure(store, new_run, records) -> None:
    run = new_run()
    with pytest.raises(PrimaryFailureError):
        store.fail_run(run.pipeline_run_id, records, actor=ACTOR)
    assert store.get_run(run.pipeline_run_id).status is R.RUNNING
    assert store.get_failures(run.pipeline_run_id) == []


def test_fail_run_persists_failures_closes_tasks_and_writes_metrics(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    retried = store.start_task(rid, "transform", actor=ACTOR)
    store.finish_task(retried.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR)
    open_task = store.start_task(rid, "transform", actor=ACTOR)
    other = store.start_task(rid, "load", actor=ACTOR)
    failed = store.fail_run(rid, [failure(True, task_id=open_task.pipeline_task_run_id),
                                  failure(False, task_id=retried.pipeline_task_run_id, message="first try")],
                            actor=ACTOR)
    assert failed.status is R.FAILED and failed.ended_at_ms is not None
    primary, secondary = store.get_failures(rid)
    assert primary.is_primary and not secondary.is_primary and secondary.error_message == "first try"
    statuses = {(t.task_name, t.task_attempt): t.status for t in store.get_tasks(rid)}
    assert statuses == {("transform", 1): TaskStatus.FAILED, ("transform", 2): TaskStatus.CANCELLED,
                        ("load", 1): TaskStatus.CANCELLED}
    assert other.pipeline_task_run_id in {e.details.get("pipeline_task_run_id") for e in store.get_events(rid)}
    m = store.get_metrics(rid)
    assert (m.task_attempt_count, m.failed_task_attempt_count, m.failed_task_count, m.task_retry_count) == (3, 1, 0, 1)


def test_fail_pending_run_has_unavailable_duration(store, new_run) -> None:
    rid = new_run(start=False).pipeline_run_id
    store.fail_run(rid, [failure(category=FailureCategory.INGESTION)], actor=ACTOR)
    m = store.get_metrics(rid)
    assert (m.duration_ms, m.duration_source, m.input_throughput_rows_per_s) == (None, DurationSource.UNAVAILABLE,
                                                                                 None)
    with pytest.raises(ObservabilityValidationError):
        store.fail_run(new_run(start=False).pipeline_run_id, [failure()], actor=ACTOR, monotonic_elapsed_ms=5)


def test_failure_cannot_reference_another_runs_task(store, new_run) -> None:
    a, b = new_run().pipeline_run_id, new_run().pipeline_run_id
    foreign_task = store.start_task(b, "extract", actor=ACTOR)
    with pytest.raises(ObservabilityValidationError, match="different pipeline run"):
        store.fail_run(a, [failure(task_id=foreign_task.pipeline_task_run_id)], actor=ACTOR)
    assert store.get_run(a).status is R.RUNNING


def test_success_never_has_a_primary_failure(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    store.complete_run(rid, actor=ACTOR)
    assert store.get_failures(rid) == []


# -------------------------------------------------------------------------------------- cancellation

def test_cancel_pending_and_running_runs(store, new_run) -> None:
    pending = store.cancel_run(new_run(start=False).pipeline_run_id, actor=S("dashboard.user"),
                               reason=S("nobody will run this"))
    assert pending.status is R.CANCELLED and pending.started_at_ms is None
    notes = [e for e in store.get_events(pending.pipeline_run_id) if e.event_type is EventType.NOTE]
    assert notes[0].details == {"cancel_reason": "nobody will run this"}
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    assert store.cancel_run(rid, actor=ACTOR, monotonic_elapsed_ms=7).status is R.CANCELLED
    assert store.get_tasks(rid)[0].status is TaskStatus.CANCELLED
    assert store.get_metrics(rid).duration_source is DurationSource.MONOTONIC
    assert task.pipeline_task_run_id


# ------------------------------------------------------------------------------------------- retries

def _failed_run(store, new_run) -> str:
    rid = new_run().pipeline_run_id
    store.fail_run(rid, [failure()], actor=ACTOR)
    return rid


def test_retry_creates_a_linked_pending_attempt_and_executes_nothing(store, new_run, clock) -> None:
    first = _failed_run(store, new_run)
    before = store.get_run(first)
    clock.advance(seconds=10)
    retry = store.request_retry(first, requested_by=S("operator"), reason=S("upstream fixed"))
    assert retry.status is R.PENDING and retry.trigger_type is TriggerType.RETRY
    assert (retry.attempt_number, retry.root_pipeline_run_id, retry.retry_of_pipeline_run_id) == (2, first, first)
    assert (retry.requested_by, retry.retry_reason) == ("operator", "upstream fixed")
    assert retry.started_at_ms is None and store.get_tasks(retry.pipeline_run_id) == []
    assert store.get_metrics(retry.pipeline_run_id) is None
    assert store.get_run(first) == before  # previous attempt untouched
    assert [e.event_type for e in store.get_events(retry.pipeline_run_id)] == [EventType.CREATED,
                                                                              EventType.RETRY_REQUESTED]
    assert store.get_events(first)[-1].details["retry_pipeline_run_id"] == retry.pipeline_run_id
    assert [r.attempt_number for r in store.get_attempt_chain(first)] == [1, 2]


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "SUCCESS"])
def test_retry_is_refused_unless_failed_or_cancelled(store, new_run, state: str) -> None:
    run = new_run(start=state != "PENDING")
    if state == "SUCCESS":
        store.complete_run(run.pipeline_run_id, actor=ACTOR)
    with pytest.raises(RetryNotAllowedError, match=state):
        store.request_retry(run.pipeline_run_id, requested_by=ACTOR, reason=S("why"))
    assert len(store.get_attempt_chain(run.pipeline_run_id)) == 1


def test_only_the_latest_attempt_can_be_retried_so_chains_never_fork(store, new_run) -> None:
    first = _failed_run(store, new_run)
    second = store.request_retry(first, requested_by=ACTOR, reason=S("r1"))
    with pytest.raises(RetryNotAllowedError, match="latest"):
        store.request_retry(first, requested_by=ACTOR, reason=S("fork"))
    store.cancel_run(second.pipeline_run_id, actor=ACTOR, reason=S("abandoned"))
    third = store.request_retry(second.pipeline_run_id, requested_by=ACTOR, reason=S("r2"))
    assert (third.attempt_number, third.root_pipeline_run_id, third.retry_of_pipeline_run_id) == \
           (3, first, second.pipeline_run_id)


def test_pending_retry_is_claimed_explicitly(store, new_run) -> None:
    retry = store.request_retry(_failed_run(store, new_run), requested_by=ACTOR, reason=S("go"))
    claimed = store.start_run(retry.pipeline_run_id, actor=S("pipeline.process"))
    assert claimed.status is R.RUNNING and claimed.attempt_number == 2


def test_retry_requires_a_reason_and_sanitized_requester(store, new_run) -> None:
    rid = _failed_run(store, new_run)
    with pytest.raises(ObservabilityValidationError):
        store.request_retry(rid, requested_by=ACTOR, reason=S("   "))
    with pytest.raises(ObservabilityValidationError):
        store.request_retry(rid, requested_by="raw-actor", reason=S("x"))  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------- metrics

def test_metrics_prefer_monotonic_and_compute_throughput(store, new_run, clock) -> None:
    rid = new_run().pipeline_run_id
    store.record_rows(rid, input_rows=5_000, output_rows=4_000, rows_source=RowsSource.REPORTED, actor=ACTOR)
    clock.advance(seconds=9)
    store.complete_run(rid, actor=ACTOR, monotonic_elapsed_ms=2_500)
    m = store.get_metrics(rid)
    assert (m.duration_ms, m.duration_source, m.input_throughput_rows_per_s) == (2_500, DurationSource.MONOTONIC,
                                                                                 2_000.0)
    assert (m.input_rows, m.output_rows, m.rows_source, m.metrics_version) == (5_000, 4_000, RowsSource.REPORTED, "v1")


def test_backwards_wall_clock_makes_duration_unavailable_but_run_still_completes(store, new_run,
                                                                                    clock: FixedClock) -> None:
    rid = new_run().pipeline_run_id
    store.record_rows(rid, input_rows=10, output_rows=None, rows_source=RowsSource.MEASURED, actor=ACTOR)
    clock.set_wall_ms(clock.utc_now_ms() - 60_000)
    assert store.complete_run(rid, actor=ACTOR).status is R.SUCCESS
    m = store.get_metrics(rid)
    assert (m.duration_ms, m.duration_source, m.input_throughput_rows_per_s) == (None, DurationSource.UNAVAILABLE,
                                                                                 None)


def test_zero_duration_has_no_throughput(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    store.record_rows(rid, input_rows=10, output_rows=None, rows_source=RowsSource.MEASURED, actor=ACTOR)
    store.complete_run(rid, actor=ACTOR)
    m = store.get_metrics(rid)
    assert (m.duration_ms, m.duration_source, m.input_throughput_rows_per_s) == (0, DurationSource.WALL_CLOCK, None)


def test_failed_task_count_uses_each_tasks_final_attempt(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    for name, outcomes in {"a": [TaskStatus.FAILED, TaskStatus.SUCCESS], "b": [TaskStatus.FAILED, TaskStatus.FAILED],
                           "c": [TaskStatus.SKIPPED]}.items():
        for outcome in outcomes:
            task = store.start_task(rid, name, actor=ACTOR)
            store.finish_task(task.pipeline_task_run_id, outcome, actor=ACTOR)
    store.fail_run(rid, [failure()], actor=ACTOR)
    m = store.get_metrics(rid)
    assert (m.task_attempt_count, m.failed_task_attempt_count, m.failed_task_count, m.task_retry_count) == (5, 3, 1, 2)


# --------------------------------------------------------------------------------------- persistence

def test_history_survives_reopening(store, new_run, db_path: Path, clock) -> None:
    rid = _failed_run(store, new_run)
    retry = store.request_retry(rid, requested_by=ACTOR, reason=S("again"))
    snapshot = (store.get_attempt_chain(rid), store.get_events(rid), store.get_failures(rid), store.get_metrics(rid))
    reopened = ObservabilityStore(db_path, clock=clock)
    assert (reopened.get_attempt_chain(rid), reopened.get_events(rid), reopened.get_failures(rid),
            reopened.get_metrics(rid)) == snapshot
    assert reopened.start_run(retry.pipeline_run_id, actor=ACTOR).status is R.RUNNING
