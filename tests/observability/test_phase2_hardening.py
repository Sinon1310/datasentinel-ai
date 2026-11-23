"""Phase 2 hardening: numeric PII redaction, and durable task-failure records (crash/restart, retry-success)."""

from __future__ import annotations

import decimal
import multiprocessing
import os
import re
import sqlite3
from pathlib import Path

import pytest

from datasentinel.observability import (
    FailureCategory, FixedClock, ObservabilityStore, ObservabilityValidationError,
    PipelineRunStatus, PipelineTracker, PrimaryFailureError, TaskStatus, lifecycle,
)
from datasentinel.observability.redaction import NUMBER_PLACEHOLDER, Redactor, numeric_is_sensitive
from tests.observability.conftest import ACTOR, failure

R = PipelineRunStatus
REDACTOR = Redactor()


# =============================================================================== numeric PII redaction

@pytest.mark.parametrize("key,value", [
    ("ref", 4929123456781234),                # card number under a neutral key (9+ digits)
    ("note", 415_555_0173),                   # phone number as an integer
    ("x", 78_051_120 + 100_000_000),          # SSN-shaped, 9 digits
    ("account_no", 73319026),                 # short account number under an identifier key
    ("customer_id", 58113),                   # small customer id: identifier key → redacted at any size
    ("customerId", 58113),                    # camelCase keys are tokenised
    ("phone", 4155550173),
    ("IBAN", 12345),
    ("card", 4929123456781234.0),             # integral float
    ("card", decimal.Decimal("4929123456781234")),
    ("account_number", "7331-9026"),          # digit-only string under an identifier key
    ("accounts", 73319026),                   # plural identifier keys match their singular
    ("customerPhones", 4155550173),
    ("ref", -4929123456781234),
])
def test_sensitive_numeric_identifiers_are_redacted(key: str, value) -> None:
    assert REDACTOR.sanitize_mapping({key: value}).value[key] == NUMBER_PLACEHOLDER


@pytest.mark.parametrize("key,value", [
    ("rows", 470_012_931), ("row_count", 470_012_931), ("rowCount", 470_012_931), ("input_rows", 5_000_000_000),
    ("bytes", 9_876_543_210), ("duration_ms", 1_763_900_000_000), ("created_at", 1_763_900_000_000),
    ("timestamp", 1_763_900_000), ("batch_size", 500), ("version", 3), ("attempt", 2), ("port", 5432),
    ("customer_count", 900_000_000), ("score", 83.5), ("amount", 1234.56), ("small", 42), ("ratio", 0.25),
    ("lat", 51.5072), ("matched_link", 1), ("total", 12_000_000_000),
])
def test_legitimate_metrics_and_counts_are_preserved(key: str, value) -> None:
    assert REDACTOR.sanitize_mapping({key: value}).value[key] == value


def test_list_items_inherit_their_key_and_nesting_uses_the_inner_key() -> None:
    value = REDACTOR.sanitize_mapping({"customer_ids": [1, 2, 3], "row_counts": [10, 20],
                                       "batch": {"rows": 999_999_999, "card_number": 4111}}).value
    assert value == {"customer_ids": ["<number>"] * 3, "row_counts": [10, 20],
                     "batch": {"rows": 999_999_999, "card_number": "<number>"}}


def test_numeric_rule_is_deterministic_and_boolean_safe() -> None:
    assert numeric_is_sensitive(99_999_999, "x") is False and numeric_is_sensitive(100_000_000, "x") is True
    assert REDACTOR.sanitize_mapping({"customer_flag": True, "phone_verified": False}).value == \
           {"customer_flag": True, "phone_verified": False}
    assert REDACTOR.sanitize_mapping({"customer_id": 58113}) == REDACTOR.sanitize_mapping({"customer_id": 58113})


# ============================================================================= durable task failures

@pytest.fixture
def tracker(store: ObservabilityStore, clock: FixedClock) -> PipelineTracker:
    return PipelineTracker(store, clock=clock)


def test_task_failure_is_persisted_immediately_while_the_run_is_open(tracker, store) -> None:
    handle = tracker.start_run("orders.daily")
    with pytest.raises(TimeoutError):
        with handle.task("fetch"):
            raise TimeoutError("upstream timed out for card 4929123456781234")
    assert store.get_run(handle.pipeline_run_id).status is R.RUNNING
    [record] = store.get_failures(handle.pipeline_run_id)
    task = store.get_tasks(handle.pipeline_run_id)[0]
    assert not record.is_primary and record.pipeline_task_run_id == task.pipeline_task_run_id
    assert (record.stage, record.category, record.error_type) == ("fetch", FailureCategory.RESOURCE_RUNTIME,
                                                                 "builtins.TimeoutError")
    assert "4929123456781234" not in record.error_message and task.status is TaskStatus.FAILED


def test_retry_success_keeps_attempt_failures_non_primary(tracker, store) -> None:
    seen_while_running = []
    with tracker.track_run("orders.daily") as run:
        for attempt in range(3):
            try:
                with run.task("fetch"):
                    if attempt < 2:
                        raise ConnectionError(f"attempt {attempt} refused")
                break
            except ConnectionError:
                seen_while_running.append(len(store.get_failures(run.pipeline_run_id)))
    assert seen_while_running == [1, 2]  # durable before the run finished
    failures = store.get_failures(run.pipeline_run_id)
    assert store.get_run(run.pipeline_run_id).status is R.SUCCESS
    assert len(failures) == 2 and not any(f.is_primary for f in failures)
    assert sorted(f.exception_chain == [] for f in failures) == [True, True]
    m = store.get_metrics(run.pipeline_run_id)
    assert (m.task_retry_count, m.failed_task_attempt_count, m.failed_task_count) == (2, 2, 0)


def test_run_failure_has_exactly_one_primary_linked_to_the_failing_task(tracker, store) -> None:
    with pytest.raises(ValueError):
        with tracker.track_run("orders.daily") as run:
            try:
                with run.task("fetch"):
                    raise ConnectionError("first attempt")
            except ConnectionError:
                pass
            with run.task("transform"):
                raise ValueError("bad cast")
    failures = store.get_failures(run.pipeline_run_id)
    primaries = [f for f in failures if f.is_primary]
    assert len(primaries) == 1 and primaries[0].stage == "transform"
    transform_task = [t for t in store.get_tasks(run.pipeline_run_id) if t.task_name == "transform"][0]
    assert primaries[0].pipeline_task_run_id == transform_task.pipeline_task_run_id
    assert sorted((f.stage, f.is_primary) for f in failures) == [("fetch", False), ("transform", False),
                                                                 ("transform", True)]


def test_task_failure_record_is_atomic_with_the_failed_status(tracker, store, db_path: Path) -> None:
    handle = tracker.start_run("orders.daily")
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute("CREATE TRIGGER inject BEFORE INSERT ON pipeline_failures BEGIN SELECT RAISE(ABORT, 'x'); END")
    try:
        with pytest.raises(KeyError) as info:
            with handle.task("fetch"):
                raise KeyError("k")
    finally:
        raw.execute("DROP TRIGGER inject")
        raw.close()
    assert any("could not record" in note for note in info.value.__notes__)
    # neither the FAILED status nor the failure record was written
    assert store.get_tasks(handle.pipeline_run_id)[0].status is TaskStatus.RUNNING
    assert store.get_failures(handle.pipeline_run_id) == []


def test_store_validates_task_failure_records(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "fetch", actor=ACTOR)
    other = store.start_task(rid, "other", actor=ACTOR)
    with pytest.raises(ObservabilityValidationError, match="FAILED"):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR, failure=failure(False))
    with pytest.raises(ObservabilityValidationError, match="different task"):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR,
                          failure=failure(False, task_id=other.pipeline_task_run_id))
    with pytest.raises(PrimaryFailureError):
        store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR, failure=failure(True))
    assert store.get_tasks(rid)[0].status is TaskStatus.RUNNING and store.get_failures(rid) == []
    store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR, failure=failure(False))
    [record] = store.get_failures(rid)
    assert record.pipeline_task_run_id == task.pipeline_task_run_id and record.stage == "fetch"
    # the run can still fail with exactly one primary; non-primary task records do not count
    store.finish_task(other.pipeline_task_run_id, TaskStatus.SUCCESS, actor=ACTOR)
    with pytest.raises(PrimaryFailureError):
        store.fail_run(rid, [failure(False)], actor=ACTOR)
    store.fail_run(rid, [failure(True, task_id=task.pipeline_task_run_id)], actor=ACTOR)
    assert sum(f.is_primary for f in store.get_failures(rid)) == 1


def test_task_failure_records_are_immutable_after_the_fact(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "fetch", actor=ACTOR)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR, failure=failure(False))
    with pytest.raises(sqlite3.IntegrityError, match="DS_IMMUTABLE"):
        raw.execute("UPDATE pipeline_failures SET is_primary = 1 WHERE pipeline_run_id = ?", (rid,))


# ------------------------------------------------------------------------------------ crash / restart

def _crashing_pipeline(db: str, id_file: str) -> None:  # pragma: no cover - runs in a child process
    from datasentinel.observability import ObservabilityStore as Store, PipelineTracker as Tracker
    tracker = Tracker(Store(db))
    handle = tracker.start_run("orders.daily", parameters={"rows": 1000, "customer_id": 58113})
    for attempt in range(2):
        try:
            with handle.task("fetch"):
                raise ConnectionError(f"attempt {attempt}: refused by api_key=gsk_" + "CrashCanary0123456789")
        except ConnectionError:
            pass
    Path(id_file).write_text(handle.pipeline_run_id)
    os._exit(1)  # hard crash: no finally blocks, no completion, no cleanup


def test_task_failure_details_survive_a_process_crash(db_path: Path, store, clock) -> None:
    id_file = db_path.parent.parent / "crashed_run_id.txt"
    child = multiprocessing.get_context("spawn").Process(target=_crashing_pipeline, args=(str(db_path), str(id_file)))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 1
    run_id = id_file.read_text()

    restarted = ObservabilityStore(db_path, clock=clock)  # a fresh process/view after the crash
    run = restarted.get_run(run_id)
    assert run.status is R.RUNNING and run.parameters == {"rows": 1000, "customer_id": "<number>"}
    tasks = restarted.get_tasks(run_id)
    assert [(t.task_attempt, t.status) for t in tasks] == [(1, TaskStatus.FAILED), (2, TaskStatus.FAILED)]
    failures = restarted.get_failures(run_id)
    assert len(failures) == 2 and not any(f.is_primary for f in failures)
    assert {f.pipeline_task_run_id for f in failures} == {t.pipeline_task_run_id for t in tasks}
    assert all(f.error_type == "builtins.ConnectionError" and "CrashCanary" not in f.error_message
               for f in failures)
    assert restarted.get_metrics(run_id) is None
    # the orphaned run is only *displayed* as stale once the threshold passes; its stored status is unchanged
    assert lifecycle.is_stale(run.status, heartbeat_at_ms=run.heartbeat_at_ms, started_at_ms=run.started_at_ms,
                              now_ms=run.started_at_ms + 900_001, stale_after_ms=900_000)
    assert restarted.get_run(run_id).status is R.RUNNING


# =================================================================================== numeric leak scan

NUMERIC_CANARIES = {
    "card": 4929123456781234, "msisdn": 4155550173, "national_id": 178051120, "account_no": 73319026,
    "customer_id": 58113, "ref": 6011000990139424,
}


def _contains_number(blob: bytes, number: int) -> bool:
    return re.search(rb"(?<![0-9A-Za-z])" + str(number).encode() + rb"(?![0-9])", blob) is not None


def test_numeric_identifiers_never_reach_the_database(store, clock, db_path: Path) -> None:
    tracker = PipelineTracker(store, clock=clock)
    params = {**NUMERIC_CANARIES, "rows": 470_012_931, "batch": {"customer_ids": [58113, 58114]},
              "card_float": float(NUMERIC_CANARIES["card"])}
    with pytest.raises(ValueError):
        with tracker.track_run("leak.numeric", parameters=params, metadata={"accounts": [73319026]}) as run:
            with run.task("charge"):
                raise ValueError(f"charge declined for {NUMERIC_CANARIES['card']} / {NUMERIC_CANARIES['msisdn']}")
    blob = b"".join(p.read_bytes() for p in db_path.parent.iterdir() if p.is_file())
    with sqlite3.connect(db_path) as conn:
        blob += "\n".join(conn.iterdump()).encode()
    assert _contains_number(blob, 470_012_931)  # the scan sees stored data: legitimate metrics survive
    leaked = [name for name, number in NUMERIC_CANARIES.items() if _contains_number(blob, number)]
    assert leaked == []
    assert _contains_number(blob, 58114) is False
