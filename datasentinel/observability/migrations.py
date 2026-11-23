"""Versioned, forward-only schema migrations with checksums.

Rules:

* Migrations are applied in version order inside one ``BEGIN IMMEDIATE`` transaction, so two
  processes initialising the same database concurrently cannot both apply them, and a failure
  leaves the database exactly as it was.
* Each applied migration's SHA-256 checksum is recorded. If the recorded checksum differs from the
  code's (an applied migration was edited), opening the database is refused.
* A database whose schema is newer than this code is refused; it is never silently downgraded.
* A non-empty SQLite file that is not an observability database is refused rather than modified.

The v1 schema enforces the important invariants in the database itself (CHECK constraints, a
partial unique index and triggers), so they hold even for writes that bypass the Python API.
Trigger error messages start with a ``DS_<CODE>:`` prefix that the store maps to typed errors.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasentinel.observability.clock import Clock
from datasentinel.observability.errors import MigrationError, SchemaVersionError

_ID32 = "length({col}) = 32 AND {col} NOT GLOB '*[^0-9a-f]*'"
_JSON_OBJECT = "json_valid({col}) AND json_type({col}) = 'object' AND length({col}) <= 65536"
_JSON_ARRAY = "json_valid({col}) AND json_type({col}) = 'array' AND length({col}) <= 65536"

_V1_SQL = f"""
CREATE TABLE pipelines (
  pipeline_id   TEXT PRIMARY KEY CHECK (length(pipeline_id) BETWEEN 1 AND 64),
  display_name  TEXT NOT NULL CHECK (length(display_name) BETWEEN 1 AND 200),
  description   TEXT CHECK (description IS NULL OR length(description) <= 2000),
  owner         TEXT CHECK (owner IS NULL OR length(owner) <= 200),
  tags_json     TEXT NOT NULL DEFAULT '[]' CHECK ({_JSON_ARRAY.format(col="tags_json")}),
  is_internal   INTEGER NOT NULL DEFAULT 0 CHECK (is_internal IN (0, 1)),
  is_synthetic  INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
  created_at_ms INTEGER NOT NULL
);

CREATE TABLE pipeline_runs (
  pipeline_run_id          TEXT PRIMARY KEY CHECK ({_ID32.format(col="pipeline_run_id")}),
  pipeline_id              TEXT NOT NULL REFERENCES pipelines (pipeline_id),
  root_pipeline_run_id     TEXT NOT NULL REFERENCES pipeline_runs (pipeline_run_id),
  retry_of_pipeline_run_id TEXT UNIQUE REFERENCES pipeline_runs (pipeline_run_id),
  attempt_number           INTEGER NOT NULL CHECK (attempt_number >= 1),
  status                   TEXT NOT NULL
                           CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED')),
  status_changed_by        TEXT NOT NULL CHECK (length(status_changed_by) BETWEEN 1 AND 128),
  status_changed_at_ms     INTEGER NOT NULL,
  trigger_type             TEXT NOT NULL
                           CHECK (trigger_type IN ('manual', 'schedule', 'api', 'retry', 'internal', 'synthetic')),
  requested_by             TEXT CHECK (requested_by IS NULL OR length(requested_by) BETWEEN 1 AND 128),
  retry_reason             TEXT CHECK (retry_reason IS NULL OR length(retry_reason) <= 1024),
  created_at_ms            INTEGER NOT NULL,
  started_at_ms            INTEGER,
  ended_at_ms              INTEGER,
  monotonic_elapsed_ms     INTEGER CHECK (monotonic_elapsed_ms IS NULL OR monotonic_elapsed_ms >= 0),
  heartbeat_at_ms          INTEGER,
  input_rows               INTEGER CHECK (input_rows IS NULL OR input_rows >= 0),
  output_rows              INTEGER CHECK (output_rows IS NULL OR output_rows >= 0),
  rows_source              TEXT CHECK (rows_source IS NULL OR rows_source IN ('measured', 'reported')),
  parameters_json          TEXT NOT NULL DEFAULT '{{}}' CHECK ({_JSON_OBJECT.format(col="parameters_json")}),
  metadata_json            TEXT NOT NULL DEFAULT '{{}}' CHECK ({_JSON_OBJECT.format(col="metadata_json")}),
  code_version             TEXT CHECK (code_version IS NULL OR length(code_version) BETWEEN 1 AND 64),
  is_synthetic             INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
  UNIQUE (root_pipeline_run_id, attempt_number),
  CHECK ((attempt_number = 1 AND retry_of_pipeline_run_id IS NULL AND root_pipeline_run_id = pipeline_run_id)
      OR (attempt_number > 1 AND retry_of_pipeline_run_id IS NOT NULL
          AND root_pipeline_run_id <> pipeline_run_id AND retry_of_pipeline_run_id <> pipeline_run_id)),
  CHECK ((status IN ('SUCCESS', 'FAILED', 'CANCELLED')) = (ended_at_ms IS NOT NULL)),
  CHECK (status IN ('PENDING', 'FAILED', 'CANCELLED') OR started_at_ms IS NOT NULL),
  CHECK (status <> 'PENDING' OR (started_at_ms IS NULL AND heartbeat_at_ms IS NULL)),
  CHECK ((input_rows IS NULL AND output_rows IS NULL) OR rows_source IS NOT NULL),
  CHECK (trigger_type <> 'synthetic' OR is_synthetic = 1)
);

CREATE TABLE pipeline_task_runs (
  pipeline_task_run_id TEXT PRIMARY KEY CHECK ({_ID32.format(col="pipeline_task_run_id")}),
  pipeline_run_id      TEXT NOT NULL REFERENCES pipeline_runs (pipeline_run_id),
  task_name            TEXT NOT NULL CHECK (length(task_name) BETWEEN 1 AND 128),
  task_attempt         INTEGER NOT NULL DEFAULT 1 CHECK (task_attempt >= 1),
  status               TEXT NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED', 'SKIPPED', 'CANCELLED')),
  started_at_ms        INTEGER NOT NULL,
  ended_at_ms          INTEGER,
  monotonic_elapsed_ms INTEGER CHECK (monotonic_elapsed_ms IS NULL OR monotonic_elapsed_ms >= 0),
  input_rows           INTEGER CHECK (input_rows IS NULL OR input_rows >= 0),
  output_rows          INTEGER CHECK (output_rows IS NULL OR output_rows >= 0),
  rows_source          TEXT CHECK (rows_source IS NULL OR rows_source IN ('measured', 'reported')),
  UNIQUE (pipeline_run_id, task_name, task_attempt),
  CHECK ((status = 'RUNNING') = (ended_at_ms IS NULL)),
  CHECK ((input_rows IS NULL AND output_rows IS NULL) OR rows_source IS NOT NULL)
);

CREATE TABLE pipeline_failures (
  pipeline_failure_id  TEXT PRIMARY KEY CHECK ({_ID32.format(col="pipeline_failure_id")}),
  pipeline_run_id      TEXT NOT NULL REFERENCES pipeline_runs (pipeline_run_id),
  pipeline_task_run_id TEXT REFERENCES pipeline_task_runs (pipeline_task_run_id),
  is_primary           INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
  category             TEXT NOT NULL CHECK (category IN ('SCHEMA', 'DATA_QUALITY', 'INGESTION', 'TRANSFORMATION',
                                                         'RESOURCE_RUNTIME', 'UNKNOWN')),
  category_source      TEXT NOT NULL CHECK (category_source IN ('classifier', 'reported')),
  classifier_rule      TEXT NOT NULL CHECK (length(classifier_rule) BETWEEN 1 AND 64),
  classifier_version   TEXT NOT NULL CHECK (length(classifier_version) BETWEEN 1 AND 64),
  error_type           TEXT NOT NULL CHECK (length(error_type) BETWEEN 1 AND 200),
  error_message        TEXT NOT NULL CHECK (length(error_message) <= 2048),
  exception_chain_json TEXT NOT NULL DEFAULT '[]' CHECK ({_JSON_ARRAY.format(col="exception_chain_json")}),
  stack_json           TEXT NOT NULL DEFAULT '[]' CHECK ({_JSON_ARRAY.format(col="stack_json")}),
  stage                TEXT CHECK (stage IS NULL OR length(stage) BETWEEN 1 AND 128),
  context_json         TEXT NOT NULL DEFAULT '{{}}' CHECK ({_JSON_OBJECT.format(col="context_json")}),
  occurred_at_ms       INTEGER NOT NULL
);
CREATE UNIQUE INDEX ux_pipeline_primary_failure ON pipeline_failures (pipeline_run_id) WHERE is_primary = 1;

CREATE TABLE pipeline_run_metrics (
  pipeline_run_id             TEXT PRIMARY KEY REFERENCES pipeline_runs (pipeline_run_id),
  metrics_version             TEXT NOT NULL,
  duration_ms                 INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
  duration_source             TEXT NOT NULL CHECK (duration_source IN ('monotonic', 'wall_clock', 'unavailable')),
  input_rows                  INTEGER CHECK (input_rows IS NULL OR input_rows >= 0),
  output_rows                 INTEGER CHECK (output_rows IS NULL OR output_rows >= 0),
  rows_source                 TEXT CHECK (rows_source IS NULL OR rows_source IN ('measured', 'reported')),
  input_throughput_rows_per_s REAL CHECK (input_throughput_rows_per_s IS NULL OR input_throughput_rows_per_s >= 0),
  task_attempt_count          INTEGER NOT NULL CHECK (task_attempt_count >= 0),
  failed_task_attempt_count   INTEGER NOT NULL CHECK (failed_task_attempt_count >= 0),
  failed_task_count           INTEGER NOT NULL CHECK (failed_task_count >= 0),
  task_retry_count            INTEGER NOT NULL CHECK (task_retry_count >= 0),
  computed_at_ms              INTEGER NOT NULL,
  CHECK ((duration_source = 'unavailable') = (duration_ms IS NULL))
);

CREATE TABLE pipeline_run_events (
  event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  pipeline_run_id TEXT NOT NULL REFERENCES pipeline_runs (pipeline_run_id),
  at_ms           INTEGER NOT NULL,
  event_type      TEXT NOT NULL CHECK (event_type IN ('created', 'status_change', 'retry_requested', 'task_started',
                                                      'task_finished', 'rows_recorded', 'note')),
  from_status     TEXT,
  to_status       TEXT,
  actor           TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 128),
  details_json    TEXT NOT NULL DEFAULT '{{}}' CHECK ({_JSON_OBJECT.format(col="details_json")})
);

CREATE INDEX ix_pr_pipeline_created ON pipeline_runs (pipeline_id, created_at_ms DESC);
CREATE INDEX ix_pr_status_created ON pipeline_runs (status, created_at_ms DESC);
CREATE INDEX ix_pr_created ON pipeline_runs (created_at_ms DESC);
CREATE INDEX ix_pr_running_hb ON pipeline_runs (heartbeat_at_ms) WHERE status = 'RUNNING';
CREATE INDEX ix_ptr_run ON pipeline_task_runs (pipeline_run_id, task_name, task_attempt);
CREATE INDEX ix_pf_run ON pipeline_failures (pipeline_run_id);
CREATE INDEX ix_pf_category_time ON pipeline_failures (category, occurred_at_ms DESC);
CREATE INDEX ix_pre_run ON pipeline_run_events (pipeline_run_id, event_id);

-- ---------------------------------------------------------------- pipelines: registered once, never changed
CREATE TRIGGER pl_no_update BEFORE UPDATE ON pipelines
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipelines cannot be modified'); END;
CREATE TRIGGER pl_no_delete BEFORE DELETE ON pipelines
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipelines cannot be deleted'); END;

-- ---------------------------------------------------------------- pipeline_runs: creation and retry lineage
CREATE TRIGGER pr_insert_initial_status BEFORE INSERT ON pipeline_runs
WHEN NEW.status NOT IN ('PENDING', 'RUNNING')
BEGIN SELECT RAISE(ABORT, 'DS_TRANSITION: pipeline runs must be created PENDING or RUNNING'); END;

CREATE TRIGGER pr_insert_synthetic_matches BEFORE INSERT ON pipeline_runs
WHEN NEW.is_synthetic IS NOT (SELECT p.is_synthetic FROM pipelines p WHERE p.pipeline_id = NEW.pipeline_id)
 AND EXISTS (SELECT 1 FROM pipelines p WHERE p.pipeline_id = NEW.pipeline_id)
BEGIN SELECT RAISE(ABORT, 'DS_INTEGRITY: run synthetic flag must match its pipeline'); END;

CREATE TRIGGER pr_retry_must_be_pending BEFORE INSERT ON pipeline_runs
WHEN NEW.attempt_number > 1 AND (NEW.status <> 'PENDING' OR NEW.trigger_type <> 'retry')
BEGIN SELECT RAISE(ABORT, 'DS_RETRY: a retry attempt must be created PENDING with trigger_type retry'); END;

CREATE TRIGGER pr_retry_valid BEFORE INSERT ON pipeline_runs
WHEN NEW.attempt_number > 1 AND NOT EXISTS (
  SELECT 1 FROM pipeline_runs p
  WHERE p.pipeline_run_id = NEW.retry_of_pipeline_run_id
    AND p.pipeline_id = NEW.pipeline_id
    AND p.root_pipeline_run_id = NEW.root_pipeline_run_id
    AND p.attempt_number = NEW.attempt_number - 1
    AND p.status IN ('FAILED', 'CANCELLED'))
BEGIN SELECT RAISE(ABORT, 'DS_RETRY: retry must follow the latest FAILED or CANCELLED attempt of the same chain'); END;

-- ---------------------------------------------------------------- pipeline_runs: lifecycle
CREATE TRIGGER pr_terminal_immutable BEFORE UPDATE ON pipeline_runs
WHEN OLD.status IN ('SUCCESS', 'FAILED', 'CANCELLED')
BEGIN SELECT RAISE(ABORT, 'DS_TERMINAL: terminal pipeline runs are immutable'); END;

CREATE TRIGGER pr_identity_immutable BEFORE UPDATE ON pipeline_runs
WHEN OLD.status NOT IN ('SUCCESS', 'FAILED', 'CANCELLED') AND (
     NEW.pipeline_run_id IS NOT OLD.pipeline_run_id OR NEW.pipeline_id IS NOT OLD.pipeline_id
  OR NEW.root_pipeline_run_id IS NOT OLD.root_pipeline_run_id
  OR NEW.retry_of_pipeline_run_id IS NOT OLD.retry_of_pipeline_run_id
  OR NEW.attempt_number IS NOT OLD.attempt_number OR NEW.trigger_type IS NOT OLD.trigger_type
  OR NEW.requested_by IS NOT OLD.requested_by OR NEW.retry_reason IS NOT OLD.retry_reason
  OR NEW.created_at_ms IS NOT OLD.created_at_ms OR NEW.parameters_json IS NOT OLD.parameters_json
  OR NEW.code_version IS NOT OLD.code_version OR NEW.is_synthetic IS NOT OLD.is_synthetic
  OR (OLD.started_at_ms IS NOT NULL AND NEW.started_at_ms IS NOT OLD.started_at_ms))
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipeline run identity and lineage cannot change'); END;

CREATE TRIGGER pr_valid_transition BEFORE UPDATE OF status ON pipeline_runs
WHEN NEW.status IS NOT OLD.status AND OLD.status NOT IN ('SUCCESS', 'FAILED', 'CANCELLED') AND NOT (
     (OLD.status = 'PENDING' AND NEW.status IN ('RUNNING', 'FAILED', 'CANCELLED'))
  OR (OLD.status = 'RUNNING' AND NEW.status IN ('SUCCESS', 'FAILED', 'CANCELLED')))
BEGIN SELECT RAISE(ABORT, 'DS_TRANSITION: pipeline run status transition is not allowed'); END;

CREATE TRIGGER pr_no_active_tasks_when_terminal BEFORE UPDATE OF status ON pipeline_runs
WHEN NEW.status IN ('SUCCESS', 'FAILED', 'CANCELLED') AND OLD.status NOT IN ('SUCCESS', 'FAILED', 'CANCELLED')
 AND EXISTS (SELECT 1 FROM pipeline_task_runs t WHERE t.pipeline_run_id = NEW.pipeline_run_id AND t.status = 'RUNNING')
BEGIN SELECT RAISE(ABORT, 'DS_ACTIVE_TASKS: a pipeline run cannot end while tasks are RUNNING'); END;

CREATE TRIGGER pr_failed_requires_primary BEFORE UPDATE OF status ON pipeline_runs
WHEN NEW.status = 'FAILED' AND OLD.status IN ('PENDING', 'RUNNING')
 AND (SELECT count(*) FROM pipeline_failures f
      WHERE f.pipeline_run_id = NEW.pipeline_run_id AND f.is_primary = 1) <> 1
BEGIN SELECT RAISE(ABORT, 'DS_PRIMARY: a FAILED pipeline run requires exactly one primary failure'); END;

CREATE TRIGGER pr_primary_only_when_failed BEFORE UPDATE OF status ON pipeline_runs
WHEN NEW.status IN ('SUCCESS', 'CANCELLED') AND OLD.status IN ('PENDING', 'RUNNING')
 AND EXISTS (SELECT 1 FROM pipeline_failures f WHERE f.pipeline_run_id = NEW.pipeline_run_id AND f.is_primary = 1)
BEGIN SELECT RAISE(ABORT, 'DS_PRIMARY: SUCCESS and CANCELLED pipeline runs cannot have a primary failure'); END;

CREATE TRIGGER pr_status_event AFTER UPDATE OF status ON pipeline_runs
WHEN NEW.status IS NOT OLD.status
BEGIN
  INSERT INTO pipeline_run_events (pipeline_run_id, at_ms, event_type, from_status, to_status, actor, details_json)
  VALUES (NEW.pipeline_run_id, NEW.status_changed_at_ms, 'status_change', OLD.status, NEW.status,
          NEW.status_changed_by, '{{}}');
END;

CREATE TRIGGER pr_no_delete BEFORE DELETE ON pipeline_runs
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipeline runs cannot be deleted'); END;

-- ---------------------------------------------------------------- pipeline_task_runs
CREATE TRIGGER ptr_insert_parent_running BEFORE INSERT ON pipeline_task_runs
BEGIN
  SELECT RAISE(ABORT, 'DS_TERMINAL: tasks cannot be added to a terminal pipeline run')
  WHERE (SELECT r.status FROM pipeline_runs r WHERE r.pipeline_run_id = NEW.pipeline_run_id)
        IN ('SUCCESS', 'FAILED', 'CANCELLED');
  SELECT RAISE(ABORT, 'DS_TRANSITION: tasks can only be started while the pipeline run is RUNNING')
  WHERE (SELECT r.status FROM pipeline_runs r WHERE r.pipeline_run_id = NEW.pipeline_run_id) = 'PENDING';
  SELECT RAISE(ABORT, 'DS_TRANSITION: task runs must be created RUNNING')
  WHERE NEW.status <> 'RUNNING';
  SELECT RAISE(ABORT, 'DS_TRANSITION: a new task attempt requires the previous attempt to have FAILED')
  WHERE NEW.task_attempt > 1 AND NOT EXISTS (
    SELECT 1 FROM pipeline_task_runs t
    WHERE t.pipeline_run_id = NEW.pipeline_run_id AND t.task_name = NEW.task_name
      AND t.task_attempt = NEW.task_attempt - 1 AND t.status = 'FAILED');
END;

CREATE TRIGGER ptr_update_guard BEFORE UPDATE ON pipeline_task_runs
BEGIN
  SELECT RAISE(ABORT, 'DS_TERMINAL: finished task runs are immutable')
  WHERE OLD.status <> 'RUNNING';
  SELECT RAISE(ABORT, 'DS_TERMINAL: tasks of a terminal pipeline run cannot change')
  WHERE (SELECT r.status FROM pipeline_runs r WHERE r.pipeline_run_id = OLD.pipeline_run_id)
        IN ('SUCCESS', 'FAILED', 'CANCELLED');
  SELECT RAISE(ABORT, 'DS_IMMUTABLE: task run identity cannot change')
  WHERE NEW.pipeline_task_run_id IS NOT OLD.pipeline_task_run_id
     OR NEW.pipeline_run_id IS NOT OLD.pipeline_run_id OR NEW.task_name IS NOT OLD.task_name
     OR NEW.task_attempt IS NOT OLD.task_attempt OR NEW.started_at_ms IS NOT OLD.started_at_ms;
END;

CREATE TRIGGER ptr_no_delete BEFORE DELETE ON pipeline_task_runs
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: task runs cannot be deleted'); END;

-- ---------------------------------------------------------------- pipeline_failures
CREATE TRIGGER pf_insert_guard BEFORE INSERT ON pipeline_failures
BEGIN
  SELECT RAISE(ABORT, 'DS_TERMINAL: failures cannot be added to a terminal pipeline run')
  WHERE (SELECT r.status FROM pipeline_runs r WHERE r.pipeline_run_id = NEW.pipeline_run_id)
        IN ('SUCCESS', 'FAILED', 'CANCELLED');
  SELECT RAISE(ABORT, 'DS_INTEGRITY: failure task must belong to the same pipeline run')
  WHERE NEW.pipeline_task_run_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM pipeline_task_runs t
    WHERE t.pipeline_task_run_id = NEW.pipeline_task_run_id AND t.pipeline_run_id = NEW.pipeline_run_id);
END;

CREATE TRIGGER pf_no_update BEFORE UPDATE ON pipeline_failures
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: failure records are immutable'); END;
CREATE TRIGGER pf_no_delete BEFORE DELETE ON pipeline_failures
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: failure records cannot be deleted'); END;

-- ---------------------------------------------------------------- pipeline_run_metrics
CREATE TRIGGER prm_requires_terminal BEFORE INSERT ON pipeline_run_metrics
WHEN (SELECT r.status FROM pipeline_runs r WHERE r.pipeline_run_id = NEW.pipeline_run_id)
     NOT IN ('SUCCESS', 'FAILED', 'CANCELLED')
BEGIN SELECT RAISE(ABORT, 'DS_TRANSITION: metrics can only be recorded for terminal pipeline runs'); END;

CREATE TRIGGER prm_no_update BEFORE UPDATE ON pipeline_run_metrics
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipeline run metrics are immutable'); END;
CREATE TRIGGER prm_no_delete BEFORE DELETE ON pipeline_run_metrics
BEGIN SELECT RAISE(ABORT, 'DS_IMMUTABLE: pipeline run metrics cannot be deleted'); END;

-- ---------------------------------------------------------------- pipeline_run_events: append-only
CREATE TRIGGER pre_no_update BEFORE UPDATE ON pipeline_run_events
BEGIN SELECT RAISE(ABORT, 'DS_APPEND_ONLY: events cannot be modified'); END;
CREATE TRIGGER pre_no_delete BEFORE DELETE ON pipeline_run_events
BEGIN SELECT RAISE(ABORT, 'DS_APPEND_ONLY: events cannot be deleted'); END;
"""


@dataclass(frozen=True)
class Migration:
    """One forward-only schema migration."""

    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """SHA-256 of the migration SQL (line endings normalised; any other edit changes it)."""
        normalised = self.sql.replace("\r\n", "\n").strip()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


MIGRATIONS: Tuple[Migration, ...] = (
    Migration(1, "initial_pipeline_observability_schema", _V1_SQL),
)
LATEST_SCHEMA_VERSION: int = MIGRATIONS[-1].version

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version       INTEGER PRIMARY KEY CHECK (version >= 1),
  name          TEXT NOT NULL,
  checksum      TEXT NOT NULL CHECK (length(checksum) = 64),
  applied_at_ms INTEGER NOT NULL
)"""


def split_statements(sql: str) -> List[str]:
    """Split a script into complete statements (trigger bodies containing ``;`` stay intact).

    Executing statements individually (instead of ``executescript``, which commits first) keeps every
    migration inside the caller's transaction.
    """
    statements: List[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        if not buffer and (not line.strip() or line.lstrip().startswith("--")):
            continue
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise MigrationError("Migration SQL ends with an incomplete statement.")
    return statements


def _validate_registry(migrations: Tuple[Migration, ...]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError("Migration versions must be contiguous and start at 1.")


def applied_migrations(conn: sqlite3.Connection) -> Dict[int, str]:
    """``{version: checksum}`` recorded in the database (empty if the table does not exist yet)."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    if not exists:
        return {}
    return {int(v): str(c) for v, c in conn.execute("SELECT version, checksum FROM schema_migrations")}


def migrate(conn: sqlite3.Connection, clock: Clock,
            migrations: Tuple[Migration, ...] = MIGRATIONS) -> int:
    """Verify recorded migrations and apply pending ones atomically. Returns the resulting version.

    ``conn`` must be in autocommit mode (``isolation_level=None``); this function owns the transaction.

    Raises:
        SchemaVersionError: the database was migrated by newer code.
        MigrationError: a checksum mismatch, a gap in recorded versions, a foreign database, or a
            failure while applying a migration (in which case nothing is changed).
    """
    _validate_registry(migrations)
    known = {m.version: m for m in migrations}
    latest = migrations[-1].version
    conn.execute("BEGIN IMMEDIATE")
    try:
        recorded = applied_migrations(conn)
        if not recorded:
            foreign = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone()[0]
            if foreign:
                raise MigrationError("The file is an existing SQLite database that is not a DataSentinel "
                                     "observability database; refusing to modify it.")
        newer = [v for v in recorded if v > latest]
        if newer:
            raise SchemaVersionError(
                f"Observability database schema version {max(newer)} is newer than this code supports "
                f"({latest}). Upgrade DataSentinel; the database is never downgraded.")
        if sorted(recorded) != list(range(1, len(recorded) + 1)):
            raise MigrationError("Recorded observability schema versions are not contiguous.")
        for version, checksum in recorded.items():
            if checksum != known[version].checksum:
                raise MigrationError(
                    f"Checksum mismatch for observability migration {version}: an applied migration was "
                    "modified. Refusing to open the database.")
        conn.execute(_MIGRATIONS_TABLE)
        current = max(recorded, default=0)
        for migration in migrations:
            if migration.version <= current:
                continue
            for statement in split_statements(migration.sql):
                conn.execute(statement)
            conn.execute("INSERT INTO schema_migrations (version, name, checksum, applied_at_ms) VALUES (?, ?, ?, ?)",
                         (migration.version, migration.name, migration.checksum, clock.utc_now_ms()))
            current = migration.version
        conn.execute("COMMIT")
        return current
    except MigrationError:
        conn.execute("ROLLBACK")
        raise
    except sqlite3.Error as exc:
        conn.execute("ROLLBACK")
        raise MigrationError(f"Applying observability migrations failed: {exc}") from exc
