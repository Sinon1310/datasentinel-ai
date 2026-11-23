"""Transactions: BEGIN IMMEDIATE, foreign keys per connection, rollback on failure, locking and concurrency."""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from pathlib import Path
from typing import Callable, List

import pytest

from datasentinel.observability import (
    IntegrityViolationError, ObservabilityDatabaseError, ObservabilityStore, PipelineRunStatus, RetryNotAllowedError,
    TaskStatus, TerminalStateError, TriggerType,
)
from datasentinel.observability import store as store_module
from tests.observability.conftest import ACTOR, SPEC, S, failure


def _snapshot(raw: sqlite3.Connection) -> tuple:
    return tuple(tuple(raw.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall())
                 for t in ("pipeline_runs", "pipeline_task_runs", "pipeline_failures", "pipeline_run_events",
                           "pipeline_run_metrics"))


# ----------------------------------------------------------------- statements issued on every connection

@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    """Record the SQL of every connection the store opens."""
    sessions: List[List[str]] = []
    original = ObservabilityStore._connect

    def _connect(self):  # type: ignore[no-untyped-def]
        conn = original(self)
        log: List[str] = []
        sessions.append(log)
        conn.set_trace_callback(log.append)
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        return conn

    monkeypatch.setattr(ObservabilityStore, "_connect", _connect)
    return sessions


def test_every_write_runs_in_one_begin_immediate_transaction(store, new_run, traced) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    store.heartbeat(rid)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR)
    store.fail_run(rid, [failure()], actor=ACTOR)
    store.request_retry(rid, requested_by=ACTOR, reason=S("again"))
    assert len(traced) == 6
    for log in traced:
        statements = [s for s in log if not s.startswith("PRAGMA")]
        assert statements[0] == "BEGIN IMMEDIATE"
        assert statements[-1] == "COMMIT"
        assert statements.count("BEGIN IMMEDIATE") == 1 and "ROLLBACK" not in statements


def test_reads_use_deferred_transactions(store, new_run, traced) -> None:
    rid = new_run().pipeline_run_id
    traced.clear()
    store.get_run(rid)
    store.get_events(rid)
    assert all([s for s in log if not s.startswith("PRAGMA")][0] == "BEGIN" for log in traced)


def test_foreign_keys_are_verified_on_every_connection(store, monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoForeignKeys:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def __getattr__(self, name: str):
            return getattr(self._conn, name)

        def execute(self, sql: str, *args):
            if sql.strip() == "PRAGMA foreign_keys":
                return self._conn.execute("SELECT 0")
            return self._conn.execute(sql, *args)

    real_connect = sqlite3.connect
    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *a, **k: _NoForeignKeys(real_connect(*a, **k)))
    with pytest.raises(ObservabilityDatabaseError, match="foreign key"):
        store.get_pipeline("orders.daily")


def test_errors_roll_back_and_are_reported_as_rollback(store, new_run, traced) -> None:
    rid = new_run().pipeline_run_id
    store.start_task(rid, "extract", actor=ACTOR)
    traced.clear()
    with pytest.raises(Exception):
        store.complete_run(rid, actor=ACTOR)
    assert [s for s in traced[0] if not s.startswith("PRAGMA")][-1] == "ROLLBACK"


# --------------------------------------------------------------------- fault injection / atomicity

def _inject(raw: sqlite3.Connection, table: str, when: str = "1") -> None:
    raw.execute(f"CREATE TRIGGER inject_fault BEFORE INSERT ON {table} WHEN {when} "
                "BEGIN SELECT RAISE(ABORT, 'injected fault'); END")


@pytest.mark.parametrize("operation", ["complete", "fail", "cancel"])
def test_terminal_transitions_are_atomic_when_metrics_insert_fails(store, new_run, raw, operation: str) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "extract", actor=ACTOR)
    if operation == "complete":
        store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    _inject(raw, "pipeline_run_metrics")
    before = _snapshot(raw)
    with pytest.raises(IntegrityViolationError, match="injected fault"):
        {"complete": lambda: store.complete_run(rid, actor=ACTOR),
         "fail": lambda: store.fail_run(rid, [failure(), failure(False)], actor=ACTOR),
         "cancel": lambda: store.cancel_run(rid, actor=ACTOR, reason=S("stop"))}[operation]()
    assert _snapshot(raw) == before  # no status change, failures, closed tasks, events or metrics persisted
    raw.execute("DROP TRIGGER inject_fault")
    assert store.get_run(rid).status is PipelineRunStatus.RUNNING
    store.cancel_run(rid, actor=ACTOR)  # the run is still usable after the rollback
    assert store.get_metrics(rid) is not None


def test_retry_request_is_atomic(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    store.fail_run(rid, [failure()], actor=ACTOR)
    _inject(raw, "pipeline_run_events", "NEW.event_type = 'retry_requested' AND NEW.pipeline_run_id = "
                                        f"'{rid}'")
    before = _snapshot(raw)
    with pytest.raises(IntegrityViolationError):
        store.request_retry(rid, requested_by=ACTOR, reason=S("x"))
    assert _snapshot(raw) == before
    raw.execute("DROP TRIGGER inject_fault")
    assert store.request_retry(rid, requested_by=ACTOR, reason=S("x")).attempt_number == 2


def test_create_run_is_atomic(store, raw) -> None:
    _inject(raw, "pipeline_run_events")
    with pytest.raises(IntegrityViolationError):
        store.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.MANUAL)
    assert raw.execute("SELECT count(*) FROM pipelines").fetchone()[0] == 0
    assert raw.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 0


# ------------------------------------------------------------------------------------- write locking

def test_writers_wait_for_the_write_lock_and_fail_cleanly(db_path: Path, store, new_run) -> None:
    rid = new_run().pipeline_run_id
    impatient = ObservabilityStore(db_path, busy_timeout_ms=50)
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ObservabilityDatabaseError, match="locked"):
            impatient.complete_run(rid, actor=ACTOR)
        assert impatient.get_run(rid).status is PipelineRunStatus.RUNNING  # WAL readers are not blocked
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert impatient.complete_run(rid, actor=ACTOR).status is PipelineRunStatus.SUCCESS


# ---------------------------------------------------------------------------------------- concurrency

def _race(n: int, action: Callable[[int], object]) -> tuple:
    barrier, results, errors = threading.Barrier(n), [], []

    def run(i: int) -> None:
        barrier.wait()
        try:
            results.append(action(i))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def test_concurrent_terminal_transitions_have_exactly_one_winner(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    actions = [lambda: store.complete_run(rid, actor=ACTOR), lambda: store.fail_run(rid, [failure()], actor=ACTOR),
               lambda: store.cancel_run(rid, actor=ACTOR)]
    results, errors = _race(12, lambda i: actions[i % 3]())
    assert len(results) == 1 and len(errors) == 11
    assert all(isinstance(e, TerminalStateError) for e in errors), errors
    final = store.get_run(rid)
    status_events = [e for e in store.get_events(rid) if e.event_type.value == "status_change"]
    assert len(status_events) == 1 and status_events[0].to_status == final.status.value
    assert len(store.get_failures(rid)) == (1 if final.status is PipelineRunStatus.FAILED else 0)


def test_concurrent_retry_requests_create_exactly_one_attempt(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    store.fail_run(rid, [failure()], actor=ACTOR)
    results, errors = _race(10, lambda i: store.request_retry(rid, requested_by=ACTOR, reason=S(f"r{i}")))
    assert len(results) == 1 and all(isinstance(e, RetryNotAllowedError) for e in errors), errors
    assert [r.attempt_number for r in store.get_attempt_chain(rid)] == [1, 2]


def test_heartbeats_racing_a_terminal_transition_never_corrupt_state(store, new_run) -> None:
    rid = new_run().pipeline_run_id

    def act(i: int):
        return store.complete_run(rid, actor=ACTOR) if i == 0 else store.heartbeat(rid)

    _results, errors = _race(8, act)
    assert all(isinstance(e, TerminalStateError) for e in errors), errors
    run = store.get_run(rid)
    assert run.status is PipelineRunStatus.SUCCESS and store.get_metrics(rid) is not None


def _process_worker(path: str, run_id: str, index: int, queue) -> None:  # pragma: no cover - runs in a child
    from datasentinel.observability import ObservabilityStore as Store
    from tests.observability.conftest import ACTOR as actor, failure as make_failure
    child = Store(path)
    try:
        if index % 2:
            child.complete_run(run_id, actor=actor)
        else:
            child.fail_run(run_id, [make_failure()], actor=actor)
        queue.put("won")
    except Exception as exc:  # noqa: BLE001
        queue.put(type(exc).__name__)


def test_concurrent_processes_have_exactly_one_winner(store, new_run, db_path: Path) -> None:
    rid = new_run().pipeline_run_id
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_process_worker, args=(str(db_path), rid, i, queue)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    outcomes = sorted(queue.get(timeout=5) for _ in procs)
    assert outcomes == ["TerminalStateError"] * 3 + ["won"]
    assert store.get_run(rid).is_terminal
