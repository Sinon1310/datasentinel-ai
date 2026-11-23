"""SQLite persistence for pipeline observability.

Design points (Milestone 3 Phase 0, approved):

* **Separate database** (``<data_dir>/observability.sqlite``), never the LangGraph checkpoint database.
* **Safe files**: the database is created atomically with mode 0600 (``O_CREAT | O_EXCL``, no symlink
  following); an existing file must be a regular file owned by the current user and is tightened to
  0600, as are its WAL/SHM side files. A directory created for it gets 0700. (POSIX only.)
* **Every connection** enables ``foreign_keys`` (verified, not assumed), a busy timeout, defensive mode
  and ``trusted_schema = OFF``. Connections are opened per operation, so the store is thread-safe.
* **Every write is one ``BEGIN IMMEDIATE`` transaction**: the write lock is taken up front, the current
  state is re-read under the lock, the change is validated against the lifecycle tables and applied.
  Any error rolls the whole operation back, so no partial lifecycle change is ever persisted.
* **Two layers of invariants**: this module validates before writing and raises precise errors; the
  schema's constraints and triggers enforce the same invariants for any write that bypasses it.
* **No raw SQL interface**: callers get typed operations only. Every statement is a fixed string with
  bound parameters; no caller-controlled value is ever interpolated into SQL.
* **Redaction before persistence**: free-form values are accepted only as
  :class:`~datasentinel.observability.models.Sanitized`.
* **Metrics** are derived here from stored facts inside the terminal transaction (callers can never
  supply them) and written once.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence

from datasentinel.observability import lifecycle
from datasentinel.observability.clock import Clock, SystemClock
from datasentinel.observability.errors import (
    ActiveTasksError, IntegrityViolationError, InvalidTransitionError, ObservabilityDatabaseError,
    ObservabilityError, ObservabilityValidationError, PipelineRunNotFoundError, PrimaryFailureError,
    RetryNotAllowedError, TerminalStateError, UnsafeDatabasePathError,
)
from datasentinel.observability.metrics import METRICS_VERSION, derive_run_metrics
from datasentinel.observability.migrations import LATEST_SCHEMA_VERSION, applied_migrations, migrate
from datasentinel.observability.models import (
    CategorySource, DurationSource, EventType, FailureCategory, FailureRecord, Pipeline, PipelineFailure,
    PipelineRun, PipelineRunEvent, PipelineRunMetrics, PipelineRunStatus, PipelineSpec, PipelineTaskRun,
    RowsSource, Sanitized, TaskStatus, TriggerType, validate_code_version, validate_non_negative,
    validate_pipeline_id, validate_pipeline_run_id, validate_task_name,
)

if TYPE_CHECKING:  # pragma: no cover
    from datasentinel.config import Settings

MAX_JSON_CHARS = 65_536
MAX_MESSAGE_CHARS = 2_048
MAX_ACTOR_CHARS = 128
MAX_REASON_CHARS = 1_024
MAX_DESCRIPTION_CHARS = 2_000
MAX_OWNER_CHARS = 200
MAX_FAILURES_PER_RUN = 50
DEFAULT_BUSY_TIMEOUT_MS = 5_000
SYSTEM_ACTOR = "datasentinel"

_TERMINAL_SQL_VALUES = tuple(s.value for s in lifecycle.TERMINAL_RUN_STATUSES)
_IS_POSIX = os.name == "posix"
_SIDE_FILE_SUFFIXES = ("-wal", "-shm", "-journal")

# Trigger message prefix -> typed error (see migrations.py).
_TRIGGER_ERRORS = (
    ("DS_TERMINAL", TerminalStateError),
    ("DS_ACTIVE_TASKS", ActiveTasksError),
    ("DS_TRANSITION", InvalidTransitionError),
    ("DS_PRIMARY", PrimaryFailureError),
    ("DS_RETRY", RetryNotAllowedError),
    ("DS_IMMUTABLE", IntegrityViolationError),
    ("DS_APPEND_ONLY", IntegrityViolationError),
    ("DS_INTEGRITY", IntegrityViolationError),
)


# ==================================================================================================
# File safety
# ==================================================================================================


def _check_existing_file(path: Path) -> bool:
    """Validate an existing database or side file. Returns False if it does not exist."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise UnsafeDatabasePathError(f"Refusing to use a symbolic link as the observability database: {path.name}")
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeDatabasePathError(f"Observability database path is not a regular file: {path.name}")
    if _IS_POSIX:
        if info.st_uid != os.geteuid():
            raise UnsafeDatabasePathError(f"Observability database is not owned by the current user: {path.name}")
        if stat.S_IMODE(info.st_mode) != 0o600:
            os.chmod(path, 0o600)
    return True


def prepare_database_file(path: Path) -> None:
    """Create the database file safely (0600, no symlink following) or validate an existing one."""
    if path.name in ("", ":memory:") or path.name.startswith("file:"):
        raise UnsafeDatabasePathError("The observability database must be a file path.")
    parent = path.parent
    if not parent.exists():
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if _IS_POSIX:
            os.chmod(parent, 0o700)
    elif not parent.is_dir():
        raise UnsafeDatabasePathError("The observability database directory is not a directory.")
    if not _check_existing_file(path):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:  # created concurrently by another process: validate it instead
            _check_existing_file(path)
        else:
            os.close(fd)
            if _IS_POSIX:
                os.chmod(path, 0o600)
    for suffix in _SIDE_FILE_SUFFIXES:
        _check_existing_file(path.with_name(path.name + suffix))


# ==================================================================================================
# Helpers
# ==================================================================================================


def _new_id() -> str:
    return uuid.uuid4().hex


def _sanitized_text(name: str, value: Any, max_chars: int, *, required: bool = True) -> Optional[str]:
    if value is None:
        if required:
            raise ObservabilityValidationError(f"{name} is required")
        return None
    if not isinstance(value, Sanitized):
        raise ObservabilityValidationError(f"{name} must be a Sanitized value produced by the redactor")
    text = value.value
    if not isinstance(text, str):
        raise ObservabilityValidationError(f"{name} must be sanitised text")
    if len(text) > max_chars:
        raise ObservabilityValidationError(f"{name} exceeds {max_chars} characters")
    if required and not text.strip():
        raise ObservabilityValidationError(f"{name} must not be empty")
    return text


def _actor(value: Any, name: str = "actor") -> str:
    text = _sanitized_text(name, value, MAX_ACTOR_CHARS)
    assert text is not None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ObservabilityValidationError(f"{name} must not contain control characters")
    return text


def _sanitized_json(name: str, value: Any, expected: type) -> str:
    if value is None:
        return "{}" if expected is dict else "[]"
    if not isinstance(value, Sanitized):
        raise ObservabilityValidationError(f"{name} must be a Sanitized value produced by the redactor")
    if not isinstance(value.value, expected):
        raise ObservabilityValidationError(f"{name} must be a sanitised {expected.__name__}")
    encoded = value.to_json()
    if len(encoded) > MAX_JSON_CHARS:
        raise ObservabilityValidationError(f"{name} exceeds {MAX_JSON_CHARS} characters once encoded")
    return encoded


def _encode_failures(failures: Sequence[Any], *, allow_primary: bool = True) -> list:
    """Validate failure records and encode their sanitised payloads (before any transaction starts)."""
    failures = list(failures)
    if len(failures) > MAX_FAILURES_PER_RUN:
        raise ObservabilityValidationError(f"At most {MAX_FAILURES_PER_RUN} failures can be recorded per run.")
    if any(not isinstance(f, FailureRecord) for f in failures):
        raise ObservabilityValidationError("failures must be FailureRecord instances.")
    if not allow_primary and any(f.is_primary for f in failures):
        raise PrimaryFailureError("SUCCESS and CANCELLED pipeline runs cannot have a primary failure.")
    return [(f, _sanitized_text("message", f.message, MAX_MESSAGE_CHARS, required=False) or "",
             _sanitized_json("exception_chain", f.exception_chain, list),
             _sanitized_json("stack", f.stack, list),
             _sanitized_json("context", f.context, dict)) for f in failures]


def _internal_json(details: dict) -> str:
    """Event details built by the store itself from ids, enums and counts (never user content)."""
    return json.dumps(details, sort_keys=True, separators=(",", ":"))


def _translate(exc: sqlite3.Error) -> ObservabilityError:
    message = str(exc)
    for prefix, error in _TRIGGER_ERRORS:
        if prefix in message:
            return error(message.split(":", 1)[-1].strip())
    if isinstance(exc, sqlite3.IntegrityError):
        return IntegrityViolationError(f"The observability database rejected the write ({message}).")
    return ObservabilityDatabaseError(f"Observability database error: {message}")


# ---------------------------------------------------------------------------------------------- rows


def _opt_enum(enum: Any, value: Optional[str]) -> Any:
    return None if value is None else enum(value)


def _row_to_pipeline(row: sqlite3.Row) -> Pipeline:
    return Pipeline(row["pipeline_id"], row["display_name"], row["description"], row["owner"],
                    tuple(json.loads(row["tags_json"])), bool(row["is_internal"]), bool(row["is_synthetic"]),
                    row["created_at_ms"])


def _row_to_run(row: sqlite3.Row) -> PipelineRun:
    return PipelineRun(
        pipeline_run_id=row["pipeline_run_id"], pipeline_id=row["pipeline_id"],
        root_pipeline_run_id=row["root_pipeline_run_id"], retry_of_pipeline_run_id=row["retry_of_pipeline_run_id"],
        attempt_number=row["attempt_number"], status=PipelineRunStatus(row["status"]),
        status_changed_by=row["status_changed_by"], status_changed_at_ms=row["status_changed_at_ms"],
        trigger_type=TriggerType(row["trigger_type"]), requested_by=row["requested_by"],
        retry_reason=row["retry_reason"], created_at_ms=row["created_at_ms"], started_at_ms=row["started_at_ms"],
        ended_at_ms=row["ended_at_ms"], monotonic_elapsed_ms=row["monotonic_elapsed_ms"],
        heartbeat_at_ms=row["heartbeat_at_ms"], input_rows=row["input_rows"], output_rows=row["output_rows"],
        rows_source=_opt_enum(RowsSource, row["rows_source"]), parameters=json.loads(row["parameters_json"]),
        metadata=json.loads(row["metadata_json"]), code_version=row["code_version"],
        is_synthetic=bool(row["is_synthetic"]),
    )


def _row_to_task(row: sqlite3.Row) -> PipelineTaskRun:
    return PipelineTaskRun(row["pipeline_task_run_id"], row["pipeline_run_id"], row["task_name"],
                           row["task_attempt"], TaskStatus(row["status"]), row["started_at_ms"], row["ended_at_ms"],
                           row["monotonic_elapsed_ms"], row["input_rows"], row["output_rows"],
                           _opt_enum(RowsSource, row["rows_source"]))


def _row_to_failure(row: sqlite3.Row) -> PipelineFailure:
    return PipelineFailure(
        row["pipeline_failure_id"], row["pipeline_run_id"], row["pipeline_task_run_id"], bool(row["is_primary"]),
        FailureCategory(row["category"]), CategorySource(row["category_source"]), row["classifier_rule"],
        row["classifier_version"], row["error_type"], row["error_message"], json.loads(row["exception_chain_json"]),
        json.loads(row["stack_json"]), row["stage"], json.loads(row["context_json"]), row["occurred_at_ms"])


def _row_to_event(row: sqlite3.Row) -> PipelineRunEvent:
    return PipelineRunEvent(row["event_id"], row["pipeline_run_id"], row["at_ms"], EventType(row["event_type"]),
                            row["from_status"], row["to_status"], row["actor"], json.loads(row["details_json"]))


def _row_to_metrics(row: sqlite3.Row) -> PipelineRunMetrics:
    return PipelineRunMetrics(
        row["pipeline_run_id"], row["metrics_version"], row["duration_ms"], DurationSource(row["duration_source"]),
        row["input_rows"], row["output_rows"], _opt_enum(RowsSource, row["rows_source"]),
        row["input_throughput_rows_per_s"], row["task_attempt_count"], row["failed_task_attempt_count"],
        row["failed_task_count"], row["task_retry_count"], row["computed_at_ms"])


# ==================================================================================================
# Store
# ==================================================================================================


class ObservabilityStore:
    """Typed, transactional access to one observability database. The only write path to it."""

    def __init__(self, path: Path | str, *, clock: Optional[Clock] = None,
                 busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS) -> None:
        if not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 0:
            raise ObservabilityValidationError("busy_timeout_ms must be a non-negative integer")
        self._path = Path(path).expanduser().absolute()
        self._clock: Clock = clock or SystemClock()
        self._busy_timeout_ms = busy_timeout_ms
        prepare_database_file(self._path)
        conn = self._connect()
        try:
            self._enable_wal(conn)
            self._schema_version = migrate(conn, self._clock)
        except sqlite3.Error as exc:
            raise _translate(exc) from exc
        finally:
            conn.close()
        for suffix in _SIDE_FILE_SUFFIXES:
            _check_existing_file(self._path.with_name(self._path.name + suffix))

    def _enable_wal(self, conn: sqlite3.Connection) -> None:
        """Switch to WAL once. The switch ignores the busy handler, so concurrent first opens retry briefly."""
        deadline = time.monotonic() + max(self._busy_timeout_ms, 1_000) / 1000
        delay = 0.005
        while True:
            try:
                if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
                    return
                mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).lower() == "wal":
                    return
                raise ObservabilityDatabaseError("Could not enable WAL journaling for the observability database.")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) and "busy" not in str(exc):
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    @classmethod
    def from_settings(cls, settings: Optional["Settings"] = None, *, demo: bool = False,
                      clock: Optional[Clock] = None) -> "ObservabilityStore":
        """Open the configured real (or synthetic demo) observability database."""
        from datasentinel.config import get_settings

        settings = settings or get_settings()
        path = settings.observability_demo_db_path if demo else settings.observability_db_path
        return cls(path, clock=clock)

    # ------------------------------------------------------------------------------------- basics

    @property
    def path(self) -> Path:
        """Absolute path of the database file."""
        return self._path

    @property
    def schema_version(self) -> int:
        """Schema version after migrations were verified and applied."""
        return self._schema_version

    def _connect(self) -> sqlite3.Connection:
        _check_existing_file(self._path)
        conn = sqlite3.connect(str(self._path), timeout=self._busy_timeout_ms / 1000, isolation_level=None,
                               check_same_thread=True)
        try:
            if hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE"):
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
            conn.execute("PRAGMA foreign_keys = ON")
            if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise ObservabilityDatabaseError("SQLite foreign key enforcement could not be enabled.")
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")
            conn.execute("PRAGMA trusted_schema = OFF")
            conn.row_factory = sqlite3.Row
        except BaseException:
            conn.close()
            raise
        return conn

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        except sqlite3.Error as exc:
            raise _translate(exc) from exc
        finally:
            conn.close()

    def _now(self) -> int:
        return int(self._clock.utc_now_ms())

    # ------------------------------------------------------------------------------- lookups (txn)

    @staticmethod
    def _load_run(conn: sqlite3.Connection, pipeline_run_id: str) -> PipelineRun:
        validate_pipeline_run_id(pipeline_run_id)
        row = conn.execute("SELECT * FROM pipeline_runs WHERE pipeline_run_id = ?", (pipeline_run_id,)).fetchone()
        if row is None:
            raise PipelineRunNotFoundError("Pipeline run not found.")
        return _row_to_run(row)

    @staticmethod
    def _load_tasks(conn: sqlite3.Connection, pipeline_run_id: str) -> List[PipelineTaskRun]:
        rows = conn.execute("SELECT * FROM pipeline_task_runs WHERE pipeline_run_id = ? "
                            "ORDER BY started_at_ms, task_name, task_attempt", (pipeline_run_id,)).fetchall()
        return [_row_to_task(r) for r in rows]

    @staticmethod
    def _require_running(run: PipelineRun, what: str) -> None:
        if run.is_terminal:
            raise TerminalStateError(f"Cannot {what}: the pipeline run is {run.status.value} (terminal).")
        if run.status is not PipelineRunStatus.RUNNING:
            raise InvalidTransitionError(f"Cannot {what}: the pipeline run is {run.status.value}, not RUNNING.")

    def _ensure_pipeline(self, conn: sqlite3.Connection, spec: PipelineSpec, now: int) -> None:
        row = conn.execute("SELECT * FROM pipelines WHERE pipeline_id = ?", (spec.pipeline_id,)).fetchone()
        if row is not None:
            if bool(row["is_synthetic"]) != spec.is_synthetic or bool(row["is_internal"]) != spec.is_internal:
                raise ObservabilityValidationError(
                    "Pipeline is already registered with different internal/synthetic flags.")
            return
        conn.execute(
            "INSERT INTO pipelines (pipeline_id, display_name, description, owner, tags_json, is_internal, "
            "is_synthetic, created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (spec.pipeline_id, spec.display_name,
             _sanitized_text("description", spec.description, MAX_DESCRIPTION_CHARS, required=False),
             _sanitized_text("owner", spec.owner, MAX_OWNER_CHARS, required=False),
             json.dumps(list(spec.tags)), int(spec.is_internal), int(spec.is_synthetic), now))

    @staticmethod
    def _event(conn: sqlite3.Connection, run_id: str, at_ms: int, event_type: EventType, actor: str,
               details: str = "{}") -> None:
        conn.execute("INSERT INTO pipeline_run_events (pipeline_run_id, at_ms, event_type, actor, details_json) "
                     "VALUES (?, ?, ?, ?, ?)", (run_id, at_ms, event_type.value, actor, details))

    @staticmethod
    def _set_status(conn: sqlite3.Connection, run: PipelineRun, target: PipelineRunStatus, actor: str, now: int,
                    *, started_at_ms: Optional[int] = None, ended_at_ms: Optional[int] = None,
                    monotonic_elapsed_ms: Optional[int] = None) -> None:
        """Compare-and-set status change; the status_change event is written by a database trigger."""
        lifecycle.validate_run_transition(run.status, target)
        cursor = conn.execute(
            "UPDATE pipeline_runs SET status = ?, status_changed_by = ?, status_changed_at_ms = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?), ended_at_ms = ?, "
            "monotonic_elapsed_ms = COALESCE(?, monotonic_elapsed_ms) "
            "WHERE pipeline_run_id = ? AND status = ?",
            (target.value, actor, now, started_at_ms, ended_at_ms, monotonic_elapsed_ms, run.pipeline_run_id,
             run.status.value))
        if cursor.rowcount != 1:  # impossible under BEGIN IMMEDIATE; kept as a guard
            raise InvalidTransitionError("The pipeline run changed concurrently; the transition was not applied.")

    def _close_running_tasks(self, conn: sqlite3.Connection, run_id: str, actor: str, now: int) -> None:
        running = conn.execute("SELECT pipeline_task_run_id, task_name, task_attempt FROM pipeline_task_runs "
                               "WHERE pipeline_run_id = ? AND status = 'RUNNING'", (run_id,)).fetchall()
        for row in running:
            conn.execute("UPDATE pipeline_task_runs SET status = 'CANCELLED', ended_at_ms = ? "
                         "WHERE pipeline_task_run_id = ? AND status = 'RUNNING'", (now, row["pipeline_task_run_id"]))
            self._event(conn, run_id, now, EventType.TASK_FINISHED, actor, _internal_json(
                {"pipeline_task_run_id": row["pipeline_task_run_id"], "task_name": row["task_name"],
                 "task_attempt": row["task_attempt"], "status": TaskStatus.CANCELLED.value,
                 "reason": "pipeline_run_ended"}))

    def _write_metrics(self, conn: sqlite3.Connection, run_id: str, now: int) -> None:
        run = self._load_run(conn, run_id)
        m = derive_run_metrics(run, self._load_tasks(conn, run_id), now)
        conn.execute(
            "INSERT INTO pipeline_run_metrics (pipeline_run_id, metrics_version, duration_ms, duration_source, "
            "input_rows, output_rows, rows_source, input_throughput_rows_per_s, task_attempt_count, "
            "failed_task_attempt_count, failed_task_count, task_retry_count, computed_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (m.pipeline_run_id, m.metrics_version, m.duration_ms, m.duration_source.value, m.input_rows,
             m.output_rows, None if m.rows_source is None else m.rows_source.value, m.input_throughput_rows_per_s,
             m.task_attempt_count, m.failed_task_attempt_count, m.failed_task_count, m.task_retry_count,
             m.computed_at_ms))

    @staticmethod
    def _insert_failures(conn: sqlite3.Connection, pipeline_run_id: str, encoded: list, now: int) -> None:
        for failure, message, chain, stack, context in encoded:
            if failure.pipeline_task_run_id is not None and not conn.execute(
                    "SELECT 1 FROM pipeline_task_runs WHERE pipeline_task_run_id = ? AND pipeline_run_id = ?",
                    (failure.pipeline_task_run_id, pipeline_run_id)).fetchone():
                raise ObservabilityValidationError("A failure references a task of a different pipeline run.")
            conn.execute(
                "INSERT INTO pipeline_failures (pipeline_failure_id, pipeline_run_id, pipeline_task_run_id, "
                "is_primary, category, category_source, classifier_rule, classifier_version, error_type, "
                "error_message, exception_chain_json, stack_json, stage, context_json, occurred_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_new_id(), pipeline_run_id, failure.pipeline_task_run_id, int(failure.is_primary),
                 failure.category.value, failure.category_source.value, failure.classifier_rule,
                 failure.classifier_version, failure.error_type, message, chain, stack, failure.stage, context,
                 now))

    @staticmethod
    def _check_elapsed(run: PipelineRun, monotonic_elapsed_ms: Optional[int]) -> Optional[int]:
        validate_non_negative("monotonic_elapsed_ms", monotonic_elapsed_ms)
        if monotonic_elapsed_ms is not None and run.status is not PipelineRunStatus.RUNNING:
            raise ObservabilityValidationError("monotonic_elapsed_ms applies only to a run that was RUNNING.")
        return monotonic_elapsed_ms

    # ===================================================================================== writes

    def register_pipeline(self, spec: PipelineSpec) -> Pipeline:
        """Register a pipeline if it does not exist yet (registrations are immutable)."""
        with self._transaction(write=True) as conn:
            self._ensure_pipeline(conn, spec, self._now())
            return _row_to_pipeline(conn.execute("SELECT * FROM pipelines WHERE pipeline_id = ?",
                                                 (spec.pipeline_id,)).fetchone())

    def create_run(self, spec: PipelineSpec, *, actor: Sanitized[str], trigger_type: TriggerType,
                   start: bool = False, requested_by: Optional[Sanitized[str]] = None,
                   parameters: Optional[Sanitized[dict]] = None, metadata: Optional[Sanitized[dict]] = None,
                   code_version: Optional[str] = None, pipeline_run_id: Optional[str] = None) -> PipelineRun:
        """Create attempt 1 of a pipeline run, PENDING (or RUNNING when ``start``), with its ``created`` event.

        Retry attempts are created only by :meth:`request_retry`.
        """
        trigger = TriggerType(trigger_type)
        if trigger is TriggerType.RETRY:
            raise ObservabilityValidationError("Retry attempts are created with request_retry().")
        if trigger is TriggerType.SYNTHETIC and not spec.is_synthetic:
            raise ObservabilityValidationError("trigger_type 'synthetic' requires a synthetic pipeline.")
        actor_text = _actor(actor)
        requested = None if requested_by is None else _actor(requested_by, "requested_by")
        params_json = _sanitized_json("parameters", parameters, dict)
        meta_json = _sanitized_json("metadata", metadata, dict)
        if code_version is not None:
            validate_code_version(code_version)
        run_id = validate_pipeline_run_id(pipeline_run_id) if pipeline_run_id is not None else _new_id()
        status = PipelineRunStatus.RUNNING if start else PipelineRunStatus.PENDING
        with self._transaction(write=True) as conn:
            now = self._now()
            self._ensure_pipeline(conn, spec, now)
            conn.execute(
                "INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, attempt_number, "
                "status, status_changed_by, status_changed_at_ms, trigger_type, requested_by, created_at_ms, "
                "started_at_ms, parameters_json, metadata_json, code_version, is_synthetic) "
                "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, spec.pipeline_id, run_id, status.value, actor_text, now, trigger.value, requested, now,
                 now if start else None, params_json, meta_json, code_version, int(spec.is_synthetic)))
            self._event(conn, run_id, now, EventType.CREATED, actor_text,
                        _internal_json({"initial_status": status.value, "trigger_type": trigger.value}))
            return self._load_run(conn, run_id)

    def start_run(self, pipeline_run_id: str, *, actor: Sanitized[str]) -> PipelineRun:
        """PENDING → RUNNING (also how a pending retry attempt is claimed by its own pipeline)."""
        actor_text = _actor(actor)
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            now = self._now()
            self._set_status(conn, run, PipelineRunStatus.RUNNING, actor_text, now, started_at_ms=now)
            return self._load_run(conn, pipeline_run_id)

    def heartbeat(self, pipeline_run_id: str) -> None:
        """Record a sign of life for a RUNNING run (no event is written, to keep the log small)."""
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            self._require_running(run, "record a heartbeat")
            conn.execute("UPDATE pipeline_runs SET heartbeat_at_ms = ? WHERE pipeline_run_id = ? AND status = 'RUNNING'",
                         (self._now(), pipeline_run_id))

    def record_rows(self, pipeline_run_id: str, *, input_rows: Optional[int], output_rows: Optional[int],
                    rows_source: RowsSource, actor: Sanitized[str]) -> PipelineRun:
        """Record run-level row counts (facts, with their source) while the run is RUNNING."""
        validate_non_negative("input_rows", input_rows)
        validate_non_negative("output_rows", output_rows)
        source = RowsSource(rows_source)
        actor_text = _actor(actor)
        if input_rows is None and output_rows is None:
            raise ObservabilityValidationError("At least one of input_rows and output_rows is required.")
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            self._require_running(run, "record row counts")
            now = self._now()
            conn.execute("UPDATE pipeline_runs SET input_rows = COALESCE(?, input_rows), "
                         "output_rows = COALESCE(?, output_rows), rows_source = ? WHERE pipeline_run_id = ?",
                         (input_rows, output_rows, source.value, pipeline_run_id))
            self._event(conn, pipeline_run_id, now, EventType.ROWS_RECORDED, actor_text, _internal_json(
                {"input_rows": input_rows, "output_rows": output_rows, "rows_source": source.value}))
            return self._load_run(conn, pipeline_run_id)

    def start_task(self, pipeline_run_id: str, task_name: str, *, actor: Sanitized[str]) -> PipelineTaskRun:
        """Start the next attempt of a task. A new attempt is allowed only after the previous one FAILED."""
        validate_task_name(task_name)
        actor_text = _actor(actor)
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            self._require_running(run, "start a task")
            last = conn.execute("SELECT task_attempt, status FROM pipeline_task_runs WHERE pipeline_run_id = ? "
                                "AND task_name = ? ORDER BY task_attempt DESC LIMIT 1",
                                (pipeline_run_id, task_name)).fetchone()
            if last is not None and last["status"] != TaskStatus.FAILED.value:
                raise InvalidTransitionError(
                    f"Task attempt {last['task_attempt']} is {last['status']}; a new attempt requires it to have FAILED.")
            attempt = 1 if last is None else last["task_attempt"] + 1
            task_id, now = _new_id(), self._now()
            conn.execute("INSERT INTO pipeline_task_runs (pipeline_task_run_id, pipeline_run_id, task_name, "
                         "task_attempt, status, started_at_ms) VALUES (?, ?, ?, ?, 'RUNNING', ?)",
                         (task_id, pipeline_run_id, task_name, attempt, now))
            self._event(conn, pipeline_run_id, now, EventType.TASK_STARTED, actor_text, _internal_json(
                {"pipeline_task_run_id": task_id, "task_name": task_name, "task_attempt": attempt}))
            return _row_to_task(conn.execute("SELECT * FROM pipeline_task_runs WHERE pipeline_task_run_id = ?",
                                             (task_id,)).fetchone())

    def finish_task(self, pipeline_task_run_id: str, status: TaskStatus, *, actor: Sanitized[str],
                    input_rows: Optional[int] = None, output_rows: Optional[int] = None,
                    rows_source: Optional[RowsSource] = None,
                    monotonic_elapsed_ms: Optional[int] = None,
                    failure: Optional[FailureRecord] = None) -> PipelineTaskRun:
        """RUNNING → SUCCESS | FAILED | SKIPPED | CANCELLED. Finished tasks never change again.

        ``failure`` (FAILED only) is the task attempt's NON-primary failure record. It is written in the same
        transaction, so the details survive a crash even if the run never finishes.
        """
        validate_pipeline_run_id(pipeline_task_run_id)
        target = TaskStatus(status)
        encoded: list = []
        if failure is not None:
            if target is not TaskStatus.FAILED:
                raise ObservabilityValidationError("A failure record can only accompany a FAILED task.")
            if failure.pipeline_task_run_id not in (None, pipeline_task_run_id):
                raise ObservabilityValidationError("The failure record references a different task.")
            encoded = _encode_failures([failure], allow_primary=False)
        actor_text = _actor(actor)
        for name, value in (("input_rows", input_rows), ("output_rows", output_rows),
                            ("monotonic_elapsed_ms", monotonic_elapsed_ms)):
            validate_non_negative(name, value)
        has_rows = input_rows is not None or output_rows is not None
        if has_rows != (rows_source is not None):
            raise ObservabilityValidationError("rows_source is required exactly when row counts are given.")
        with self._transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM pipeline_task_runs WHERE pipeline_task_run_id = ?",
                               (pipeline_task_run_id,)).fetchone()
            if row is None:
                raise PipelineRunNotFoundError("Task run not found.")
            task = _row_to_task(row)
            run = self._load_run(conn, task.pipeline_run_id)
            if run.is_terminal:
                raise TerminalStateError(f"The pipeline run is {run.status.value}; its tasks can no longer change.")
            lifecycle.validate_task_transition(task.status, target)
            now = self._now()
            encoded = [(dataclasses.replace(f, pipeline_task_run_id=pipeline_task_run_id, stage=f.stage or task.task_name),
                        *rest) for f, *rest in encoded]
            self._insert_failures(conn, task.pipeline_run_id, encoded, now)
            conn.execute("UPDATE pipeline_task_runs SET status = ?, ended_at_ms = ?, monotonic_elapsed_ms = ?, "
                         "input_rows = ?, output_rows = ?, rows_source = ? "
                         "WHERE pipeline_task_run_id = ? AND status = 'RUNNING'",
                         (target.value, now, monotonic_elapsed_ms, input_rows, output_rows,
                          None if rows_source is None else RowsSource(rows_source).value, pipeline_task_run_id))
            self._event(conn, task.pipeline_run_id, now, EventType.TASK_FINISHED, actor_text, _internal_json(
                {"pipeline_task_run_id": pipeline_task_run_id, "task_name": task.task_name,
                 "task_attempt": task.task_attempt, "status": target.value}))
            return _row_to_task(conn.execute("SELECT * FROM pipeline_task_runs WHERE pipeline_task_run_id = ?",
                                             (pipeline_task_run_id,)).fetchone())

    def complete_run(self, pipeline_run_id: str, *, actor: Sanitized[str],
                     monotonic_elapsed_ms: Optional[int] = None,
                     failures: Sequence[FailureRecord] = ()) -> PipelineRun:
        """RUNNING → SUCCESS, refusing while any task is RUNNING; writes the run's metrics atomically.

        ``failures`` may carry NON-primary failures (for example task attempts that failed and were
        retried successfully); they are persisted in the same transaction.
        """
        actor_text = _actor(actor)
        encoded = _encode_failures(failures, allow_primary=False)
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            lifecycle.validate_run_transition(run.status, PipelineRunStatus.SUCCESS)
            elapsed = self._check_elapsed(run, monotonic_elapsed_ms)
            active = conn.execute("SELECT count(*) FROM pipeline_task_runs WHERE pipeline_run_id = ? "
                                  "AND status = 'RUNNING'", (pipeline_run_id,)).fetchone()[0]
            if active:
                raise ActiveTasksError(f"Cannot complete the pipeline run: {active} task(s) are still RUNNING.")
            if conn.execute("SELECT 1 FROM pipeline_failures WHERE pipeline_run_id = ? AND is_primary = 1",
                            (pipeline_run_id,)).fetchone():
                raise PrimaryFailureError("A pipeline run with a primary failure cannot succeed.")
            now = self._now()
            self._insert_failures(conn, pipeline_run_id, encoded, now)
            self._set_status(conn, run, PipelineRunStatus.SUCCESS, actor_text, now, ended_at_ms=now,
                             monotonic_elapsed_ms=elapsed)
            self._write_metrics(conn, pipeline_run_id, now)
            return self._load_run(conn, pipeline_run_id)

    def fail_run(self, pipeline_run_id: str, failures: Sequence[FailureRecord], *, actor: Sanitized[str],
                 monotonic_elapsed_ms: Optional[int] = None) -> PipelineRun:
        """PENDING/RUNNING → FAILED with exactly one primary failure, in one transaction.

        Persists the failures, closes RUNNING tasks as CANCELLED, changes the status and writes metrics.
        """
        actor_text = _actor(actor)
        failures = list(failures)
        if not failures:
            raise PrimaryFailureError("fail_run requires at least one failure record.")
        encoded = _encode_failures(failures)
        if sum(1 for f in failures if f.is_primary) != 1:
            raise PrimaryFailureError("A FAILED pipeline run requires exactly one primary failure.")
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            lifecycle.validate_run_transition(run.status, PipelineRunStatus.FAILED)
            elapsed = self._check_elapsed(run, monotonic_elapsed_ms)
            now = self._now()
            self._insert_failures(conn, pipeline_run_id, encoded, now)
            self._close_running_tasks(conn, pipeline_run_id, actor_text, now)
            self._set_status(conn, run, PipelineRunStatus.FAILED, actor_text, now, ended_at_ms=now,
                             monotonic_elapsed_ms=elapsed)
            self._write_metrics(conn, pipeline_run_id, now)
            return self._load_run(conn, pipeline_run_id)

    def cancel_run(self, pipeline_run_id: str, *, actor: Sanitized[str], reason: Optional[Sanitized[str]] = None,
                   monotonic_elapsed_ms: Optional[int] = None,
                   failures: Sequence[FailureRecord] = (),
                   require_status: Optional[PipelineRunStatus] = None) -> PipelineRun:
        """PENDING/RUNNING → CANCELLED: closes RUNNING tasks as CANCELLED and writes metrics, atomically.

        Policy (decision I4) — who may cancel a RUNNING run — is enforced by the callers above the store;
        ``require_status`` lets them make that check atomically under the write lock.
        """
        actor_text = _actor(actor)
        reason_text = _sanitized_text("reason", reason, MAX_REASON_CHARS, required=False)
        encoded = _encode_failures(failures, allow_primary=False)
        with self._transaction(write=True) as conn:
            run = self._load_run(conn, pipeline_run_id)
            if require_status is not None and run.status is not PipelineRunStatus(require_status):
                lifecycle.validate_run_transition(run.status, PipelineRunStatus.CANCELLED)  # terminal → TerminalStateError
                raise InvalidTransitionError(
                    f"Only {PipelineRunStatus(require_status).value} runs can be cancelled here "
                    f"(this run is {run.status.value}).")
            lifecycle.validate_run_transition(run.status, PipelineRunStatus.CANCELLED)
            elapsed = self._check_elapsed(run, monotonic_elapsed_ms)
            now = self._now()
            if reason_text:
                self._event(conn, pipeline_run_id, now, EventType.NOTE, actor_text,
                            json.dumps({"cancel_reason": reason_text}, sort_keys=True, ensure_ascii=False))
            self._insert_failures(conn, pipeline_run_id, encoded, now)
            self._close_running_tasks(conn, pipeline_run_id, actor_text, now)
            self._set_status(conn, run, PipelineRunStatus.CANCELLED, actor_text, now, ended_at_ms=now,
                             monotonic_elapsed_ms=elapsed)
            self._write_metrics(conn, pipeline_run_id, now)
            return self._load_run(conn, pipeline_run_id)

    def request_retry(self, pipeline_run_id: str, *, requested_by: Sanitized[str],
                      reason: Sanitized[str]) -> PipelineRun:
        """Record-only retry: create attempt n+1 as PENDING. Nothing is executed.

        Allowed only for the latest attempt of its chain, and only when it is FAILED or CANCELLED. The
        previous attempt is not modified (it is immutable); ``retry_requested`` events are appended to
        both attempts.
        """
        requester = _actor(requested_by, "requested_by")
        reason_text = _sanitized_text("reason", reason, MAX_REASON_CHARS)
        with self._transaction(write=True) as conn:
            previous = self._load_run(conn, pipeline_run_id)
            latest = conn.execute("SELECT max(attempt_number) FROM pipeline_runs WHERE root_pipeline_run_id = ?",
                                  (previous.root_pipeline_run_id,)).fetchone()[0]
            if previous.attempt_number != latest:
                raise RetryNotAllowedError("Only the latest attempt of a pipeline run can be retried.")
            if previous.status not in lifecycle.RETRYABLE_RUN_STATUSES:
                raise RetryNotAllowedError(
                    f"Retries can be requested only for FAILED or CANCELLED runs (this run is {previous.status.value}).")
            new_id, now = _new_id(), self._now()
            conn.execute(
                "INSERT INTO pipeline_runs (pipeline_run_id, pipeline_id, root_pipeline_run_id, "
                "retry_of_pipeline_run_id, attempt_number, status, status_changed_by, status_changed_at_ms, "
                "trigger_type, requested_by, retry_reason, created_at_ms, parameters_json, metadata_json, "
                "code_version, is_synthetic) "
                "SELECT ?, pipeline_id, root_pipeline_run_id, pipeline_run_id, attempt_number + 1, 'PENDING', ?, ?, "
                "'retry', ?, ?, ?, parameters_json, metadata_json, code_version, is_synthetic "
                "FROM pipeline_runs WHERE pipeline_run_id = ?",
                (new_id, requester, now, requester, reason_text, now, pipeline_run_id))
            self._event(conn, new_id, now, EventType.CREATED, requester, _internal_json(
                {"initial_status": PipelineRunStatus.PENDING.value, "trigger_type": TriggerType.RETRY.value}))
            self._event(conn, new_id, now, EventType.RETRY_REQUESTED, requester, _internal_json(
                {"retry_of_pipeline_run_id": pipeline_run_id, "attempt_number": previous.attempt_number + 1}))
            self._event(conn, pipeline_run_id, now, EventType.RETRY_REQUESTED, requester, _internal_json(
                {"retry_pipeline_run_id": new_id, "attempt_number": previous.attempt_number + 1}))
            return self._load_run(conn, new_id)

    # ====================================================================================== reads

    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        validate_pipeline_id(pipeline_id)
        with self._transaction(write=False) as conn:
            row = conn.execute("SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,)).fetchone()
            return None if row is None else _row_to_pipeline(row)

    def get_run(self, pipeline_run_id: str) -> PipelineRun:
        with self._transaction(write=False) as conn:
            return self._load_run(conn, pipeline_run_id)

    def get_tasks(self, pipeline_run_id: str) -> List[PipelineTaskRun]:
        with self._transaction(write=False) as conn:
            self._load_run(conn, pipeline_run_id)
            return self._load_tasks(conn, pipeline_run_id)

    def get_failures(self, pipeline_run_id: str) -> List[PipelineFailure]:
        with self._transaction(write=False) as conn:
            self._load_run(conn, pipeline_run_id)
            rows = conn.execute("SELECT * FROM pipeline_failures WHERE pipeline_run_id = ? "
                                "ORDER BY is_primary DESC, occurred_at_ms, pipeline_failure_id",
                                (pipeline_run_id,)).fetchall()
            return [_row_to_failure(r) for r in rows]

    def get_events(self, pipeline_run_id: str) -> List[PipelineRunEvent]:
        with self._transaction(write=False) as conn:
            self._load_run(conn, pipeline_run_id)
            rows = conn.execute("SELECT * FROM pipeline_run_events WHERE pipeline_run_id = ? ORDER BY event_id",
                                (pipeline_run_id,)).fetchall()
            return [_row_to_event(r) for r in rows]

    def get_metrics(self, pipeline_run_id: str) -> Optional[PipelineRunMetrics]:
        with self._transaction(write=False) as conn:
            self._load_run(conn, pipeline_run_id)
            row = conn.execute("SELECT * FROM pipeline_run_metrics WHERE pipeline_run_id = ?",
                               (pipeline_run_id,)).fetchone()
            return None if row is None else _row_to_metrics(row)

    def get_attempt_chain(self, pipeline_run_id: str) -> List[PipelineRun]:
        """Every attempt in the run's retry chain, ordered by attempt number."""
        with self._transaction(write=False) as conn:
            run = self._load_run(conn, pipeline_run_id)
            rows = conn.execute("SELECT * FROM pipeline_runs WHERE root_pipeline_run_id = ? ORDER BY attempt_number",
                                (run.root_pipeline_run_id,)).fetchall()
            return [_row_to_run(r) for r in rows]

    def recorded_schema_versions(self) -> List[int]:
        """Schema versions recorded in the database (for diagnostics)."""
        with self._transaction(write=False) as conn:
            return sorted(applied_migrations(conn))


__all__ = ["ObservabilityStore", "derive_run_metrics", "prepare_database_file", "METRICS_VERSION",
           "LATEST_SCHEMA_VERSION", "SYSTEM_ACTOR"]
