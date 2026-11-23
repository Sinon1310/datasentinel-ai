"""Phase 2 additions to the store, and an end-to-end secret/PII leak scan of the raw database bytes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datasentinel.observability import (
    FailureCategory, IntegrityViolationError, InvalidTransitionError, PipelineRunStatus, PipelineTracker,
    PrimaryFailureError, TaskStatus, TerminalStateError,
)
from tests.observability.conftest import ACTOR, S, failure

R = PipelineRunStatus


# ------------------------------------------------------------- non-primary failures on SUCCESS/CANCELLED

def test_complete_run_persists_non_primary_failures_atomically(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    task = store.start_task(rid, "fetch", actor=ACTOR)
    store.finish_task(task.pipeline_task_run_id, TaskStatus.FAILED, actor=ACTOR)
    store.complete_run(rid, actor=ACTOR, failures=[failure(False, task_id=task.pipeline_task_run_id)])
    [stored] = store.get_failures(rid)
    assert not stored.is_primary and stored.pipeline_task_run_id == task.pipeline_task_run_id


@pytest.mark.parametrize("operation", ["complete", "cancel"])
def test_primary_failures_are_refused_on_success_and_cancel(store, new_run, operation: str) -> None:
    rid = new_run().pipeline_run_id
    call = (lambda: store.complete_run(rid, actor=ACTOR, failures=[failure(True)])) if operation == "complete" \
        else (lambda: store.cancel_run(rid, actor=ACTOR, failures=[failure(True)]))
    with pytest.raises(PrimaryFailureError):
        call()
    assert store.get_run(rid).status is R.RUNNING and store.get_failures(rid) == []


def test_completion_with_failures_rolls_back_together(store, new_run, raw) -> None:
    rid = new_run().pipeline_run_id
    raw.execute("CREATE TRIGGER inject BEFORE INSERT ON pipeline_run_metrics BEGIN SELECT RAISE(ABORT, 'x'); END")
    with pytest.raises(IntegrityViolationError):
        store.complete_run(rid, actor=ACTOR, failures=[failure(False)])
    raw.execute("DROP TRIGGER inject")
    assert store.get_run(rid).status is R.RUNNING and store.get_failures(rid) == []


def test_cancel_require_status_is_checked_under_the_lock(store, new_run) -> None:
    running = new_run().pipeline_run_id
    with pytest.raises(InvalidTransitionError, match="Only PENDING"):
        store.cancel_run(running, actor=ACTOR, require_status=R.PENDING)
    assert store.get_run(running).status is R.RUNNING
    pending = new_run(start=False).pipeline_run_id
    assert store.cancel_run(pending, actor=ACTOR, require_status=R.PENDING).status is R.CANCELLED
    with pytest.raises(TerminalStateError):
        store.cancel_run(pending, actor=ACTOR, require_status=R.PENDING)


def test_phase_one_signatures_remain_compatible(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    assert store.complete_run(rid, actor=ACTOR).status is R.SUCCESS
    assert store.cancel_run(new_run().pipeline_run_id, actor=ACTOR, reason=S("x")).status is R.CANCELLED


# ------------------------------------------------------------------------------------ leak scan

SECRETS = [
    "gsk_" + "LeakCanaryGroqKey0123456789",
    "sk-proj-" + "LeakCanaryOpenAIKey012345",
    "AKIA" + "LEAKCANARY012345",
    "hunter2-LeakCanaryPassword",
    "LeakCanaryDsnPassword",
    "canary.person@example.com",
    "+1 415 555 0199",
    "4111111111111111",
    "eyJhbGciOiJIUzI1NiJ9.eyJsZWFrIjoiY2FuYXJ5In0.c2lnbmF0dXJlX2xlYWtfY2FuYXJ5",
]
SOURCE_MARKER = "LEAK_CANARY_SOURCE_LINE_MARKER"


def _fail_with_secrets() -> None:
    LEAK_CANARY_SOURCE_LINE_MARKER = SECRETS[3]  # noqa: N806,F841 - a local variable that must never be stored
    try:
        raise ConnectionError(f"login failed for {SECRETS[5]} using {SECRETS[0]}")
    except ConnectionError as exc:
        raise RuntimeError(f"task failed; dsn=postgres://etl:{SECRETS[4]}@db/prod token {SECRETS[1]}") from exc


def _all_bytes(db_dir: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(db_dir.iterdir()) if p.is_file())


def test_no_secret_or_pii_reaches_the_database_through_any_write_path(store, clock, db_path: Path) -> None:
    tracker = PipelineTracker(store, clock=clock)
    secret_params = {"password": SECRETS[3], "api_key": SECRETS[0], "nested": {"aws": SECRETS[2],
                     "contact": SECRETS[5], "phone": SECRETS[6], "card": SECRETS[7]}, "jwt": SECRETS[8],
                     "url": f"postgres://etl:{SECRETS[4]}@db/prod"}
    tracker.register_pipeline("leak.check", display_name=f"Leak {SECRETS[5]}", description=f"key {SECRETS[0]}",
                              owner=SECRETS[5])
    with pytest.raises(RuntimeError):
        with tracker.track_run("leak.check", parameters=secret_params, metadata=secret_params,
                               actor=f"svc {SECRETS[5]}", requested_by=SECRETS[5]) as run:
            with run.task("connect"):
                _fail_with_secrets()
    retry = tracker.request_retry(run.pipeline_run_id, requested_by=SECRETS[5],
                                  reason=f"retry with {SECRETS[0]} and password={SECRETS[3]}")
    tracker.cancel_pending_run(retry.pipeline_run_id, actor=SECRETS[5], reason=f"call {SECRETS[6]}")
    handle = tracker.start_run("leak.check")
    handle.fail(category=FailureCategory.INGESTION, message=f"reported with {SECRETS[8]} {SECRETS[7]}")

    blob = _all_bytes(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        blob += "\n".join(conn.iterdump()).encode()
    assert b"leak.check" in blob and b"<email>" in blob and b"[REDACTED]" in blob  # the scan sees stored data
    leaks = [s for s in SECRETS if s.encode() in blob]
    assert leaks == []
    assert SOURCE_MARKER.encode() not in blob  # no source lines and no local variables were stored
    failure = [f for f in store.get_failures(run.pipeline_run_id) if f.is_primary][0]
    assert failure.exception_chain[0]["type"] == "builtins.ConnectionError"
    assert any(frame["function"] == "_fail_with_secrets" for frame in failure.stack)
