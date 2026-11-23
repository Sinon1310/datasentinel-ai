"""File permissions, unsafe paths, redaction-at-the-boundary, inert SQL and the absence of a raw SQL API."""

from __future__ import annotations

import inspect
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from datasentinel.config import Settings
from datasentinel.observability import (
    ObservabilityStore, ObservabilityValidationError, PipelineSpec, TriggerType, UnsafeDatabasePathError,
)
from datasentinel.observability.models import FailureRecord
from tests.observability.conftest import ACTOR, SPEC, S, failure

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# --------------------------------------------------------------------------------------- permissions

@posix_only
def test_new_database_and_directory_are_private(store, db_path: Path) -> None:
    assert _mode(db_path) == 0o600
    assert _mode(db_path.parent) == 0o700


@posix_only
def test_wal_and_shm_files_are_private_while_in_use(store, new_run, db_path: Path) -> None:
    holder = sqlite3.connect(db_path)  # keeps the WAL/SHM files alive
    try:
        holder.execute("SELECT count(*) FROM pipeline_runs").fetchone()
        new_run()
        for suffix in ("-wal", "-shm"):
            side = db_path.with_name(db_path.name + suffix)
            assert side.exists() and _mode(side) == 0o600, suffix
    finally:
        holder.close()


@posix_only
def test_existing_loose_permissions_are_tightened(tmp_path: Path) -> None:
    path = tmp_path / "obs.sqlite"
    path.touch()
    path.chmod(0o644)
    ObservabilityStore(path)
    assert _mode(path) == 0o600


@posix_only
def test_existing_directory_permissions_are_left_alone(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    ObservabilityStore(shared / "obs.sqlite")
    assert _mode(shared) == 0o755 and _mode(shared / "obs.sqlite") == 0o600


@posix_only
def test_symlinked_database_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.sqlite"
    target.touch()
    link = tmp_path / "obs.sqlite"
    link.symlink_to(target)
    with pytest.raises(UnsafeDatabasePathError, match="symbolic link"):
        ObservabilityStore(link)
    assert target.stat().st_size == 0


@posix_only
def test_symlinked_side_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "obs.sqlite"
    (tmp_path / "victim").touch()
    (tmp_path / "obs.sqlite-wal").symlink_to(tmp_path / "victim")
    with pytest.raises(UnsafeDatabasePathError):
        ObservabilityStore(path)


@pytest.mark.parametrize("name", [":memory:", "file:obs.sqlite?mode=memory"])
def test_non_file_targets_are_refused(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UnsafeDatabasePathError):
        ObservabilityStore(name)


def test_directory_path_is_refused(tmp_path: Path) -> None:
    (tmp_path / "dir.sqlite").mkdir()
    with pytest.raises(UnsafeDatabasePathError, match="regular file"):
        ObservabilityStore(tmp_path / "dir.sqlite")


# ------------------------------------------------------------------------ redaction at the boundary

@pytest.mark.parametrize("field", ["actor", "parameters", "metadata", "requested_by"])
def test_unsanitized_free_form_values_are_rejected(store, field: str) -> None:
    kwargs = {"actor": ACTOR, "trigger_type": TriggerType.MANUAL,
              field: "api_key=gsk_should_never_be_stored" if field != "parameters" and field != "metadata"
              else {"password": "hunter2"}}
    with pytest.raises(ObservabilityValidationError, match="Sanitized"):
        store.create_run(SPEC, **kwargs)


def test_unsanitized_failure_message_is_rejected(store, new_run) -> None:
    rid = new_run().pipeline_run_id
    record = failure()
    object.__setattr__(record, "message", "raw traceback text")
    with pytest.raises(ObservabilityValidationError, match="Sanitized"):
        store.fail_run(rid, [record], actor=ACTOR)
    assert store.get_failures(rid) == []


def test_unsanitized_pipeline_description_is_rejected(store) -> None:
    spec = PipelineSpec("p1", "P1")
    object.__setattr__(spec, "description", "contains secrets")
    with pytest.raises(ObservabilityValidationError):
        store.register_pipeline(spec)


@pytest.mark.parametrize("field,value", [("actor", S("x" * 129)), ("actor", S("a\nb")), ("actor", S("  ")),
                                         ("actor", S({"not": "text"})), ("parameters", S("not a dict")),
                                         ("parameters", S({"blob": "x" * 70_000}))])
def test_sanitized_values_are_still_bounded(store, field: str, value) -> None:
    kwargs = {"actor": ACTOR, "trigger_type": TriggerType.MANUAL, field: value}
    with pytest.raises(ObservabilityValidationError):
        store.create_run(SPEC, **kwargs)


def test_oversized_failure_message_is_rejected(store, new_run) -> None:
    with pytest.raises(ObservabilityValidationError):
        store.fail_run(new_run().pipeline_run_id, [failure(message="x" * 2049)], actor=ACTOR)


def test_too_many_failures_are_rejected(store, new_run) -> None:
    with pytest.raises(ObservabilityValidationError):
        store.fail_run(new_run().pipeline_run_id, [failure()] + [failure(False)] * 50, actor=ACTOR)
    with pytest.raises(ObservabilityValidationError):
        store.fail_run(new_run().pipeline_run_id, ["not a record"], actor=ACTOR)  # type: ignore[list-item]


# -------------------------------------------------------------------------------------- inert SQL

INJECTION = "x'); DROP TABLE pipeline_runs; --"


def test_injection_text_is_stored_as_data(store, db_path: Path) -> None:
    run = store.create_run(SPEC, actor=S(INJECTION), trigger_type=TriggerType.MANUAL, start=True,
                           parameters=S({INJECTION: INJECTION}))
    store.cancel_run(run.pipeline_run_id, actor=S(INJECTION), reason=S(INJECTION))
    stored = store.get_run(run.pipeline_run_id)
    assert stored.status_changed_by == INJECTION and stored.parameters == {INJECTION: INJECTION}
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 1


@pytest.mark.parametrize("call", [
    lambda s: s.get_run(INJECTION),
    lambda s: s.get_pipeline(INJECTION),
    lambda s: s.start_task("a" * 32, INJECTION, actor=ACTOR),
    lambda s: s.create_run(PipelineSpec(INJECTION, "x"), actor=ACTOR, trigger_type=TriggerType.MANUAL),
    lambda s: s.create_run(SPEC, actor=ACTOR, trigger_type=TriggerType.MANUAL, code_version=INJECTION),
    lambda s: FailureRecord(**{**failure().__dict__, "error_type": INJECTION}),
])
def test_injection_text_in_identifiers_is_rejected_before_sql(store, call) -> None:
    with pytest.raises(ObservabilityValidationError):
        call(store)


# ---------------------------------------------------------------------------------- no raw SQL API

PUBLIC_API = {"path", "schema_version", "from_settings", "register_pipeline", "create_run", "start_run", "heartbeat",
              "record_rows", "start_task", "finish_task", "complete_run", "fail_run", "cancel_run", "request_retry",
              "get_pipeline", "get_run", "get_tasks", "get_failures", "get_events", "get_metrics",
              "get_attempt_chain", "recorded_schema_versions"}


def test_store_exposes_only_typed_operations() -> None:
    public = {name for name in dir(ObservabilityStore) if not name.startswith("_")}
    assert public == PUBLIC_API
    for name in public:
        member = getattr(ObservabilityStore, name)
        if callable(member):
            params = set(inspect.signature(member).parameters)
            assert not params & {"sql", "query", "statement", "where", "order_by", "column"}, name


def test_store_never_hands_out_a_connection(store, new_run) -> None:
    run = new_run()
    for value in list(vars(store).values()) + [store.get_run(run.pipeline_run_id)]:
        assert not isinstance(value, sqlite3.Connection)


def test_store_source_has_no_dynamic_sql_built_from_arguments() -> None:
    source = Path(inspect.getfile(ObservabilityStore)).read_text()
    assert "executescript" not in source
    # The only f-string SQL is the busy-timeout pragma, built from a validated integer.
    assert [line.strip() for line in source.splitlines() if 'execute(f"' in line] == [
        'conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")']


# ------------------------------------------------------------------------------------------ settings

def test_from_settings_uses_data_dir_and_separate_demo_file(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    real = ObservabilityStore.from_settings(settings)
    demo = ObservabilityStore.from_settings(settings, demo=True)
    assert real.path == (tmp_path / "data" / "observability.sqlite").absolute()
    assert demo.path == (tmp_path / "data" / "observability_demo.sqlite").absolute()
    assert real.path != demo.path
