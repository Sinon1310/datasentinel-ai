"""Database-level invariants, exercised with RAW SQL that bypasses the store entirely.

These prove the schema itself (constraints, partial unique index, triggers) protects the history even
from code that ignores the Python API.
"""

from __future__ import annotations

import itertools
import sqlite3
import uuid

import pytest

from datasentinel.observability import PipelineRunStatus, TaskStatus, lifecycle
from tests.observability.conftest import ACTOR, SPEC, failure

TERMINAL = ("SUCCESS", "FAILED", "CANCELLED")


def _rejects(raw: sqlite3.Connection, sql: str, params=(), match: str = "") -> str:
    with pytest.raises(sqlite3.IntegrityError) as info:
        raw.execute(sql, params)
    assert match in str(info.value), str(info.value)
    return str(info.value)


def _insert_failure(raw, run_id: str, primary: int = 1, task_id=None) -> None:
    raw.execute("INSERT INTO pipeline_failures (pipeline_failure_id, pipeline_run_id, pipeline_task_run_id, "
                "is_primary, category, category_source, classifier_rule, classifier_version, error_type, "
                "error_message, occurred_at_ms) VALUES (?, ?, ?, ?, 'UNKNOWN', 'classifier', 'r', 'v1', 'E', 'm', 0)",
                (uuid.uuid4().hex, run_id, task_id, primary))


def _in_state(store, new_run, state: str) -> str:
    rid = new_run(start=state != "PENDING").pipeline_run_id
    if state == "SUCCESS":
        store.complete_run(rid, actor=ACTOR)
    elif state == "FAILED":
        store.fail_run(rid, [failure()], actor=ACTOR)
    elif state == "CANCELLED":
        store.cancel_run(rid, actor=ACTOR)
    return rid


def _counts(raw) -> tuple:
    return tuple(raw.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                 for t in ("pipeline_runs", "pipeline_task_runs", "pipeline_failures", "pipeline_run_events",
                           "pipeline_run_metrics"))


# ------------------------------------------------------------------- transitions agree with lifecycle

@pytest.mark.parametrize("current,target",
                         [(a.value, b.value) for a, b in itertools.product(PipelineRunStatus, PipelineRunStatus)
                          if a is not b])
def test_database_transition_rules_match_the_lifecycle_table(store, new_run, raw, current: str, target: str) -> None:
    rid = _in_state(store, new_run, current)
    if target == "FAILED" and current not in TERMINAL:
        _insert_failure(raw, rid)
    ended = 1 if target in TERMINAL else None
    sql = ("UPDATE pipeline_runs SET status = ?, status_changed_by = 'raw', status_changed_at_ms = 1, "
           "started_at_ms = CASE WHEN ? = 'PENDING' THEN started_at_ms ELSE COALESCE(started_at_ms, 1) END, "
           "ended_at_ms = ? WHERE pipeline_run_id = ?")
    if lifecycle.can_transition_run(PipelineRunStatus(current), PipelineRunStatus(target)):
        raw.execute(sql, (target, target, ended, rid))
        assert raw.execute("SELECT status FROM pipeline_runs WHERE pipeline_run_id = ?", (rid,)).fetchone()[0] == target
    else:
        message = _rejects(raw, sql, (target, target, ended, rid))
        if current in TERMINAL:
            assert "DS_TERMINAL" in message


def test_status_change_events_are_generated_by_the_database(store, new_run, raw) -> None:
    rid = new_run(start=False).pipeline_run_id
    raw.execute("UPDATE pipeline_runs SET status = 'RUNNING', started_at_ms = 5, status_changed_by = 'raw.actor', "
                "status_changed_at_ms = 5 WHERE pipeline_run_id = ?", (rid,))
    row = raw.execute("SELECT event_type, from_status, to_status, actor, at_ms FROM pipeline_run_events "
                      "WHERE pipeline_run_id = ? ORDER BY event_id DESC LIMIT 1", (rid,)).fetchone()
    assert row == ("status_change", "PENDING", "RUNNING", "raw.actor", 5)


# --------------------------------------------------------------------------------- runs: immutability

def test_terminal_runs_reject_any_update(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "SUCCESS")
    for column, value in (("heartbeat_at_ms", 1), ("metadata_json", "{}"), ("input_rows", 3), ("status", "RUNNING")):
        _rejects(raw, f"UPDATE pipeline_runs SET {column} = ? WHERE pipeline_run_id = ?", (value, rid), "DS_TERMINAL")


def test_runs_cannot_be_deleted(store, new_run, raw) -> None:
    for state in ("PENDING", "SUCCESS"):
        rid = _in_state(store, new_run, state)
        _rejects(raw, "DELETE FROM pipeline_runs WHERE pipeline_run_id = ?", (rid,), "DS_IMMUTABLE")


@pytest.mark.parametrize("column,value", [("pipeline_id", "other"), ("attempt_number", 2), ("trigger_type", "api"),
                                          ("created_at_ms", 0), ("parameters_json", '{"x":1}'),
                                          ("is_synthetic", 1), ("started_at_ms", 0), ("requested_by", "mallory")])
def test_lineage_and_identity_cannot_change_on_open_runs(store, new_run, raw, column: str, value) -> None:
    rid = new_run().pipeline_run_id
    _rejects(raw, f"UPDATE pipeline_runs SET {column} = ? WHERE pipeline_run_id = ?", (value, rid))


def test_runs_must_be_created_pending_or_running(store, raw) -> None:
    store.register_pipeline(SPEC)
    rid = uuid.uuid4().hex
    _rejects(raw, "INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, attempt_number, "
                  "status, status_changed_by, status_changed_at_ms, trigger_type, created_at_ms, started_at_ms, "
                  "ended_at_ms) VALUES (?, 'orders.daily', ?, 1, 'SUCCESS', 'raw', 0, 'manual', 0, 0, 1)",
             (rid, rid), "DS_TRANSITION")


def test_foreign_keys_and_lineage_checks(store, raw) -> None:
    rid = uuid.uuid4().hex
    base = ("INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, attempt_number, status, "
            "status_changed_by, status_changed_at_ms, trigger_type, created_at_ms) VALUES (?, ?, ?, 1, 'PENDING', "
            "'raw', 0, 'manual', 0)")
    _rejects(raw, base, (rid, "no.such.pipeline", rid), "FOREIGN KEY")
    store.register_pipeline(SPEC)
    _rejects(raw, base, (rid, "orders.daily", "0" * 32), "")  # attempt 1 must be its own root
    _rejects(raw, base, ("NOT-HEX", "orders.daily", "NOT-HEX"), "CHECK")


def test_synthetic_flag_must_match_the_pipeline(store, raw) -> None:
    store.register_pipeline(SPEC)
    rid = uuid.uuid4().hex
    _rejects(raw, "INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, attempt_number, "
                  "status, status_changed_by, status_changed_at_ms, trigger_type, created_at_ms, is_synthetic) "
                  "VALUES (?, 'orders.daily', ?, 1, 'PENDING', 'raw', 0, 'manual', 0, 1)", (rid, rid), "DS_INTEGRITY")


# --------------------------------------------------------------------------------- runs: primary failure

def test_failed_requires_exactly_one_primary_failure(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    fail = ("UPDATE pipeline_runs SET status = 'FAILED', ended_at_ms = 1, status_changed_by = 'raw' "
            "WHERE pipeline_run_id = ?")
    _rejects(raw, fail, (rid,), "DS_PRIMARY")
    _insert_failure(raw, rid, primary=0)
    _rejects(raw, fail, (rid,), "DS_PRIMARY")
    _insert_failure(raw, rid, primary=1)
    _rejects(raw, "INSERT INTO pipeline_failures (pipeline_failure_id, pipeline_run_id, is_primary, category, "
                  "category_source, classifier_rule, classifier_version, error_type, error_message, occurred_at_ms) "
                  "VALUES (?, ?, 1, 'UNKNOWN', 'classifier', 'r', 'v1', 'E', 'm', 0)", (uuid.uuid4().hex, rid),
             "UNIQUE")
    raw.execute(fail, (rid,))


@pytest.mark.parametrize("target", ["SUCCESS", "CANCELLED"])
def test_success_and_cancelled_forbid_a_primary_failure(store, new_run, raw, target: str) -> None:
    rid = new_run().pipeline_run_id
    _insert_failure(raw, rid, primary=1)
    _rejects(raw, "UPDATE pipeline_runs SET status = ?, ended_at_ms = 1 WHERE pipeline_run_id = ?", (target, rid),
             "DS_PRIMARY")


def test_success_allows_non_primary_failures_from_retried_tasks(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    _insert_failure(raw, rid, primary=0)
    raw.execute("UPDATE pipeline_runs SET status = 'SUCCESS', ended_at_ms = 1 WHERE pipeline_run_id = ?", (rid,))


@pytest.mark.parametrize("target", TERMINAL)
def test_runs_cannot_end_with_running_tasks(store, new_run, raw, target: str) -> None:
    rid = new_run().pipeline_run_id
    store.start_task(rid, "extract", actor=ACTOR)
    if target == "FAILED":
        _insert_failure(raw, rid)
    _rejects(raw, "UPDATE pipeline_runs SET status = ?, ended_at_ms = 1 WHERE pipeline_run_id = ?", (target, rid),
             "DS_ACTIVE_TASKS")


# ------------------------------------------------------------------------------------- retry lineage

def _retry_insert(raw, prev_id: str, root: str, attempt: int, status: str = "PENDING", trigger: str = "retry",
                  pipeline: str = "orders.daily") -> str:
    new_id = uuid.uuid4().hex
    raw.execute("INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, "
                "retry_of_pipeline_run_id, attempt_number, status, status_changed_by, status_changed_at_ms, "
                "trigger_type, created_at_ms, started_at_ms) VALUES (?, ?, ?, ?, ?, ?, 'raw', 0, ?, 0, ?)",
                (new_id, pipeline, root, prev_id, attempt, status, trigger, 0 if status == "RUNNING" else None))
    return new_id


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "SUCCESS"])
def test_retry_of_non_retryable_attempt_is_rejected(store, new_run, raw, state: str) -> None:
    rid = _in_state(store, new_run, state)
    with pytest.raises(sqlite3.IntegrityError, match="DS_RETRY"):
        _retry_insert(raw, rid, rid, 2)


def test_retry_rules_are_enforced_by_the_database(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "FAILED")
    with pytest.raises(sqlite3.IntegrityError, match="DS_RETRY"):
        _retry_insert(raw, rid, rid, 3)  # skips an attempt number
    with pytest.raises(sqlite3.IntegrityError, match="DS_RETRY"):
        _retry_insert(raw, rid, rid, 2, status="RUNNING")  # retries are created PENDING
    with pytest.raises(sqlite3.IntegrityError, match="DS_RETRY"):
        _retry_insert(raw, rid, rid, 2, trigger="manual")
    other = _in_state(store, new_run, "FAILED")
    with pytest.raises(sqlite3.IntegrityError, match="DS_RETRY"):
        _retry_insert(raw, rid, other, 2)  # wrong root
    second = _retry_insert(raw, rid, rid, 2)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE|DS_RETRY"):
        _retry_insert(raw, rid, rid, 2)  # fork: a second retry of the same attempt


def test_unique_retry_of_blocks_forks_even_with_a_new_attempt_number(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "FAILED")
    _retry_insert(raw, rid, rid, 2)
    with pytest.raises(sqlite3.IntegrityError):
        _retry_insert(raw, rid, rid, 3)


# --------------------------------------------------------------------------------------------- tasks

def _task_insert(raw, run_id: str, name: str = "extract", attempt: int = 1, status: str = "RUNNING") -> str:
    task_id = uuid.uuid4().hex
    raw.execute("INSERT INTO pipeline_task_runs (pipeline_task_run_id, pipeline_run_id, task_name, task_attempt, "
                "status, started_at_ms, ended_at_ms) VALUES (?, ?, ?, ?, ?, 0, ?)",
                (task_id, run_id, name, attempt, status, None if status == "RUNNING" else 1))
    return task_id


def test_tasks_require_a_running_parent(store, new_run, raw) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="DS_TRANSITION"):
        _task_insert(raw, new_run(start=False).pipeline_run_id)
    for state in TERMINAL:
        with pytest.raises(sqlite3.IntegrityError, match="DS_TERMINAL"):
            _task_insert(raw, _in_state(store, new_run, state))


def test_task_creation_and_attempt_rules(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    with pytest.raises(sqlite3.IntegrityError, match="DS_TRANSITION"):
        _task_insert(raw, rid, status="SUCCESS")
    first = _task_insert(raw, rid)
    with pytest.raises(sqlite3.IntegrityError, match="DS_TRANSITION"):
        _task_insert(raw, rid, attempt=2)  # previous attempt still RUNNING
    raw.execute("UPDATE pipeline_task_runs SET status = 'FAILED', ended_at_ms = 1 WHERE pipeline_task_run_id = ?",
                (first,))
    _task_insert(raw, rid, attempt=2)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        _task_insert(raw, rid, attempt=2)


def test_finished_tasks_and_tasks_of_terminal_runs_are_immutable(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    _rejects(raw, "UPDATE pipeline_task_runs SET output_rows = 1, rows_source = 'reported' "
                  "WHERE pipeline_task_run_id = ?", (task.pipeline_task_run_id,), "DS_TERMINAL")
    _rejects(raw, "DELETE FROM pipeline_task_runs WHERE pipeline_task_run_id = ?", (task.pipeline_task_run_id,),
             "DS_IMMUTABLE")
    open_task = store.start_task(rid, "load", actor=ACTOR)
    _rejects(raw, "UPDATE pipeline_task_runs SET task_name = 'renamed' WHERE pipeline_task_run_id = ?",
             (open_task.pipeline_task_run_id,), "DS_IMMUTABLE")
    _rejects(raw, "UPDATE pipeline_task_runs SET status = 'SUCCESS' WHERE pipeline_task_run_id = ?",
             (open_task.pipeline_task_run_id,), "CHECK")  # a finished task needs ended_at_ms


# ------------------------------------------------------------------------------------------ failures

def test_failures_only_while_open_and_never_changed(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "FAILED")
    with pytest.raises(sqlite3.IntegrityError, match="DS_TERMINAL"):
        _insert_failure(raw, rid, primary=0)
    _rejects(raw, "UPDATE pipeline_failures SET error_message = 'x' WHERE pipeline_run_id = ?", (rid,),
             "DS_IMMUTABLE")
    _rejects(raw, "DELETE FROM pipeline_failures WHERE pipeline_run_id = ?", (rid,), "DS_IMMUTABLE")
    a, b = new_run().pipeline_run_id, new_run().pipeline_run_id
    foreign = store.start_task(b, "extract", actor=ACTOR)
    with pytest.raises(sqlite3.IntegrityError, match="DS_INTEGRITY"):
        _insert_failure(raw, a, primary=0, task_id=foreign.pipeline_task_run_id)


def test_failure_json_columns_are_validated(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    _rejects(raw, "INSERT INTO pipeline_failures (pipeline_failure_id, pipeline_run_id, is_primary, category, "
                  "category_source, classifier_rule, classifier_version, error_type, error_message, stack_json, "
                  "occurred_at_ms) VALUES (?, ?, 0, 'UNKNOWN', 'classifier', 'r', 'v1', 'E', 'm', 'not json', 0)",
             (uuid.uuid4().hex, rid), "CHECK")
    _rejects(raw, "INSERT INTO pipeline_failures (pipeline_failure_id, pipeline_run_id, is_primary, category, "
                  "category_source, classifier_rule, classifier_version, error_type, error_message, occurred_at_ms) "
                  "VALUES (?, ?, 0, 'MADE_UP', 'classifier', 'r', 'v1', 'E', 'm', 0)", (uuid.uuid4().hex, rid),
             "CHECK")


# ------------------------------------------------------------------------------------------- metrics

_METRICS_INSERT = ("INSERT INTO pipeline_run_metrics (pipeline_run_id, metrics_version, duration_source, "
                   "task_attempt_count, failed_task_attempt_count, failed_task_count, task_retry_count, "
                   "computed_at_ms) VALUES (?, 'v1', 'unavailable', 0, 0, 0, 0, 0)")


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_metrics_only_for_terminal_runs(store, new_run, raw, state: str) -> None:
    rid = _in_state(store, new_run, state)
    _rejects(raw, _METRICS_INSERT, (rid,), "DS_TRANSITION")


def test_metrics_are_written_once_and_immutable(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "SUCCESS")
    _rejects(raw, _METRICS_INSERT, (rid,), "UNIQUE")
    _rejects(raw, "UPDATE pipeline_run_metrics SET duration_ms = 1 WHERE pipeline_run_id = ?", (rid,), "DS_IMMUTABLE")
    _rejects(raw, "DELETE FROM pipeline_run_metrics WHERE pipeline_run_id = ?", (rid,), "DS_IMMUTABLE")


# -------------------------------------------------------------------------------------------- events

def test_events_are_append_only(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    _rejects(raw, "UPDATE pipeline_run_events SET actor = 'mallory' WHERE pipeline_run_id = ?", (rid,),
             "DS_APPEND_ONLY")
    _rejects(raw, "DELETE FROM pipeline_run_events WHERE pipeline_run_id = ?", (rid,), "DS_APPEND_ONLY")
    _rejects(raw, "INSERT INTO pipeline_run_events (pipeline_run_id, at_ms, event_type, actor) "
                  "VALUES (?, 0, 'made_up', 'x')", (rid,), "CHECK")


def test_pipelines_are_immutable(store, raw) -> None:
    store.register_pipeline(SPEC)
    _rejects(raw, "UPDATE pipelines SET is_synthetic = 1", (), "DS_IMMUTABLE")
    _rejects(raw, "DELETE FROM pipelines", (), "DS_IMMUTABLE")


def test_rejected_raw_writes_leave_no_trace(store, new_run, raw) -> None:
    rid = _in_state(store, new_run, "FAILED")
    before = _counts(raw)
    for sql in ("UPDATE pipeline_runs SET status = 'RUNNING' WHERE pipeline_run_id = ?",
                "DELETE FROM pipeline_runs WHERE pipeline_run_id = ?"):
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(sql, (rid,))
    assert _counts(raw) == before
