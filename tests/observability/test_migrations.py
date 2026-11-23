"""Versioned migrations: fresh install, idempotent reopen, checksums, newer schemas, foreign files, atomicity."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from datasentinel.observability import migrations
from datasentinel.observability.clock import FixedClock
from datasentinel.observability.errors import MigrationError, SchemaVersionError
from datasentinel.observability.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS, Migration, migrate
from datasentinel.observability.store import ObservabilityStore

EXPECTED_TABLES = {"schema_migrations", "pipelines", "pipeline_runs", "pipeline_task_runs", "pipeline_failures",
                   "pipeline_run_metrics", "pipeline_run_events"}


def _objects(path: Path, kind: str) -> set:
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'",
                                           (kind,))}


def test_fresh_database_is_created_at_the_latest_version(store: ObservabilityStore, db_path: Path) -> None:
    assert store.schema_version == LATEST_SCHEMA_VERSION == 1
    assert store.recorded_schema_versions() == [1]
    assert _objects(db_path, "table") == EXPECTED_TABLES
    assert {"ux_pipeline_primary_failure", "ix_pr_running_hb"} <= _objects(db_path, "index")
    assert {"pr_terminal_immutable", "pr_status_event", "pr_retry_valid", "pre_no_update",
            "prm_requires_terminal", "ptr_insert_parent_running"} <= _objects(db_path, "trigger")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        name, checksum = conn.execute("SELECT name, checksum FROM schema_migrations").fetchone()
    assert (name, checksum) == (MIGRATIONS[0].name, MIGRATIONS[0].checksum)


def test_reopening_is_a_no_op(store: ObservabilityStore, db_path: Path, clock: FixedClock) -> None:
    with sqlite3.connect(db_path) as conn:
        before = conn.execute("SELECT * FROM schema_migrations").fetchall()
    clock.advance(seconds=60)
    reopened = ObservabilityStore(db_path, clock=clock)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT * FROM schema_migrations").fetchall() == before
    assert reopened.schema_version == 1


def test_checksum_is_stable_and_detects_edits() -> None:
    m = MIGRATIONS[0]
    assert m.checksum == Migration(1, m.name, m.sql.replace("\n", "\r\n")).checksum
    assert m.checksum != Migration(1, m.name, m.sql + "\n-- edited").checksum
    assert len(m.checksum) == 64


def test_checksum_mismatch_is_refused(store: ObservabilityStore, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE schema_migrations SET checksum = ? WHERE version = 1", ("0" * 64,))
    with pytest.raises(MigrationError, match="Checksum mismatch"):
        ObservabilityStore(db_path)


def test_edited_migration_in_code_is_refused(store: ObservabilityStore, db_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    edited = (Migration(1, MIGRATIONS[0].name, MIGRATIONS[0].sql + "\nCREATE TABLE extra (x INTEGER);"),)
    monkeypatch.setattr(migrations, "MIGRATIONS", edited)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(MigrationError, match="Checksum mismatch"):
            migrate(conn, FixedClock(), edited)
    finally:
        conn.close()
    assert "extra" not in _objects(db_path, "table")


def test_newer_schema_is_refused_and_never_downgraded(store: ObservabilityStore, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO schema_migrations VALUES (2, 'future', ?, 0)", ("a" * 64,))
    with pytest.raises(SchemaVersionError, match="newer than this code"):
        ObservabilityStore(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 2


def test_gap_in_recorded_versions_is_refused(store: ObservabilityStore, db_path: Path) -> None:
    three = (MIGRATIONS[0], Migration(2, "second", "CREATE TABLE t2 (x INTEGER);"),
             Migration(3, "third", "CREATE TABLE t3 (x INTEGER);"))
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("INSERT INTO schema_migrations VALUES (3, 'third', ?, 0)", (three[2].checksum,))
        with pytest.raises(MigrationError, match="not contiguous"):
            migrate(conn, FixedClock(), three)
    finally:
        conn.close()


def test_foreign_sqlite_database_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER)")
    path.chmod(0o600)
    with pytest.raises(MigrationError, match="not a DataSentinel"):
        ObservabilityStore(path)
    assert _objects(path, "table") == {"customers"}


def test_a_failing_migration_leaves_the_database_unchanged(store: ObservabilityStore, db_path: Path) -> None:
    broken = (MIGRATIONS[0], Migration(2, "adds_then_fails",
                                       "CREATE TABLE added (x INTEGER);\nINSERT INTO missing_table VALUES (1);"))
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(MigrationError, match="failed"):
            migrate(conn, FixedClock(), broken)
        assert not conn.in_transaction
    finally:
        conn.close()
    assert "added" not in _objects(db_path, "table")
    with sqlite3.connect(db_path) as c:
        assert [r[0] for r in c.execute("SELECT version FROM schema_migrations")] == [1]


def test_forward_migration_is_applied_once(store: ObservabilityStore, db_path: Path) -> None:
    extended = (MIGRATIONS[0], Migration(2, "add_table", "CREATE TABLE added (x INTEGER);"))
    for _ in range(2):
        conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            assert migrate(conn, FixedClock(), extended) == 2
        finally:
            conn.close()
    with sqlite3.connect(db_path) as c:
        assert [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version")] == [1, 2]


def test_registry_must_be_contiguous(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(MigrationError, match="contiguous"):
            migrate(conn, FixedClock(), (Migration(2, "x", "SELECT 1;"),))
    finally:
        conn.close()


def test_statement_splitter_keeps_trigger_bodies_intact() -> None:
    statements = migrations.split_statements(MIGRATIONS[0].sql)
    triggers = [s for s in statements if s.startswith("CREATE TRIGGER")]
    assert len(triggers) == len(_trigger_names())
    assert all(s.endswith("END;") for s in triggers)
    with pytest.raises(MigrationError):
        migrations.split_statements("CREATE TABLE x (a INTEGER")


def _trigger_names() -> list:
    return [line.split()[2] for line in MIGRATIONS[0].sql.splitlines() if line.startswith("CREATE TRIGGER")]


def test_concurrent_initialisation_applies_migrations_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "concurrent" / "obs.sqlite"
    barrier, errors_seen = threading.Barrier(8), []

    def open_store() -> None:
        barrier.wait()
        try:
            ObservabilityStore(path)
        except Exception as exc:  # noqa: BLE001
            errors_seen.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors_seen == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1
