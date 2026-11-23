"""PipelineTracker: the writer facade for Python pipeline instrumentation.

Every value passes through the :class:`~datasentinel.observability.redaction.Redactor` before it reaches
the store; failures are classified deterministically by the
:class:`~datasentinel.observability.classifier.FailureClassifier`; durations are measured with the
monotonic clock of the process that runs the work.

Typical use::

    tracker = PipelineTracker.from_settings()
    with tracker.track_run("orders.daily", parameters={"date": "2026-11-23"}) as run:
        with run.task("extract") as task:
            rows = extract()
            task.record_rows(output_rows=len(rows), source=RowsSource.MEASURED)
        with run.task("load"):
            load(rows)

Semantics:

* Leaving ``track_run`` normally completes the run (SUCCESS); an ``Exception`` fails it with exactly one
  primary failure; other ``BaseException`` types (``KeyboardInterrupt``, ``SystemExit``) cancel it. The original exception is always
  re-raised and is never masked by a tracking error (the tracking error is attached as a note instead).
* A task that raises is recorded as a FAILED task attempt together with a NON-primary failure record, in
  one transaction, so the details survive a crash. That record stays non-primary forever (failure records
  are immutable). If the same exception then ends the run, the run gets its one primary failure, linked to
  the same task; if the task is retried successfully and the run completes, there is no primary failure.
* Retries are **record-only**: :meth:`PipelineTracker.request_retry` creates a PENDING attempt and never
  runs anything. Only the pipeline's own code runs it, by claiming it (``track_run(attach_to=...)``).
* The dashboard may cancel only PENDING attempts (:meth:`PipelineTracker.cancel_pending_run`); a RUNNING
  run can be cancelled only by the process that owns it (:meth:`RunHandle.cancel`).
* Heartbeats are throttled to ``pipeline_heartbeat_min_interval_seconds``.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, Optional, Sequence, Tuple

from datasentinel.observability.classifier import Classification, FailureClassifier, exception_chain
from datasentinel.observability.clock import Clock, SystemClock
from datasentinel.observability.errors import (
    InvalidTransitionError, ObservabilityValidationError, TerminalStateError,
)
from datasentinel.observability.models import (
    FailureCategory, FailureRecord, Pipeline, PipelineRun, PipelineRunStatus, PipelineSpec, PipelineTaskRun,
    RowsSource, TaskStatus, TriggerType, validate_pipeline_id,
)
from datasentinel.observability.redaction import MAX_REASON_CHARS, Redactor
from datasentinel.observability.store import ObservabilityStore

if TYPE_CHECKING:  # pragma: no cover
    from datasentinel.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_S = 15.0


def _elapsed_ms(clock: Clock, start_ns: int) -> int:
    return max(0, (clock.monotonic_ns() - start_ns) // 1_000_000)


class PipelineTracker:
    """Records pipeline runs, tasks, failures and retries. Never executes or re-runs anything."""

    def __init__(self, store: ObservabilityStore, *, clock: Optional[Clock] = None,
                 redactor: Optional[Redactor] = None, classifier: Optional[FailureClassifier] = None,
                 heartbeat_min_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S) -> None:
        if not isinstance(store, ObservabilityStore):
            raise ObservabilityValidationError("PipelineTracker requires an ObservabilityStore")
        if heartbeat_min_interval_s <= 0:
            raise ObservabilityValidationError("heartbeat_min_interval_s must be positive")
        self._store = store
        self._clock: Clock = clock or SystemClock()
        self._redactor = redactor or Redactor()
        self._classifier = classifier or FailureClassifier(redactor=self._redactor)
        self._heartbeat_interval_ns = int(heartbeat_min_interval_s * 1_000_000_000)

    @classmethod
    def from_settings(cls, settings: Optional["Settings"] = None, *, clock: Optional[Clock] = None,
                      classifier: Optional[FailureClassifier] = None) -> "PipelineTracker":
        """A tracker writing to the configured (real) observability database."""
        from datasentinel.config import get_settings

        settings = settings or get_settings()
        store = ObservabilityStore.from_settings(settings, clock=clock)
        return cls(store, clock=clock, classifier=classifier,
                   heartbeat_min_interval_s=float(settings.pipeline_heartbeat_min_interval_seconds))

    # ------------------------------------------------------------------------------------- helpers

    def _actor(self, actor: Any, pipeline_id: Optional[str] = None):
        return self._redactor.sanitize_actor(actor if actor else f"pipeline:{pipeline_id or 'unknown'}")

    def _spec(self, pipeline_id: str, display_name: Optional[str], *, description: Any = None, owner: Any = None,
              tags: Sequence[str] = (), is_internal: bool = False) -> PipelineSpec:
        validate_pipeline_id(pipeline_id)
        return PipelineSpec(
            pipeline_id=pipeline_id,
            display_name=" ".join(self._redactor.redact_text(display_name or pipeline_id, max_chars=200,
                                                             keep_newlines=False).split()) or pipeline_id,
            description=None if description is None else self._redactor.sanitize_text(description, max_chars=2000),
            owner=None if owner is None else self._redactor.sanitize_text(owner, max_chars=200),
            tags=tuple(tags), is_internal=is_internal)

    def build_failure(self, exc: Optional[BaseException], *, primary: bool, task: Optional[PipelineTaskRun] = None,
                      reported_category: Optional[FailureCategory] = None,
                      message: Optional[str] = None) -> FailureRecord:
        """Classify, capture and redact one failure into a store-ready record."""
        if reported_category is None and exc is not None:
            attribute = getattr(exc, "failure_category", None)
            reported_category = attribute if isinstance(attribute, FailureCategory) else None
        classification: Classification = self._classifier.classify(exc, reported_category=reported_category)
        context: dict = {"classification": classification.as_context()}
        if exc is not None:
            for link in exception_chain(exc):
                gate = getattr(link, "gate_result", None)
                if gate is not None and hasattr(gate, "to_context"):
                    context["quality_gate"] = gate.to_context()
                    break
            captured = self._redactor.capture_exception(exc)
            error_type, text, chain, stack = (captured.error_type, captured.message, captured.exception_chain,
                                              captured.stack)
        else:
            error_type, chain, stack = "datasentinel.ReportedFailure", None, None
            text = self._redactor.sanitize_text(message or "Failure reported by the pipeline.", max_chars=2048)
        return FailureRecord(
            category=classification.category, category_source=classification.category_source,
            classifier_rule=classification.rule, classifier_version=classification.version, error_type=error_type,
            message=text, is_primary=primary, exception_chain=chain, stack=stack,
            context=self._redactor.sanitize_mapping(context),
            stage=None if task is None else task.task_name,
            pipeline_task_run_id=None if task is None else task.pipeline_task_run_id)

    # --------------------------------------------------------------------------------- operations

    def register_pipeline(self, pipeline_id: str, display_name: Optional[str] = None, *, description: Any = None,
                          owner: Any = None, tags: Sequence[str] = (), is_internal: bool = False) -> Pipeline:
        """Register a pipeline (idempotent; registrations are immutable)."""
        return self._store.register_pipeline(self._spec(pipeline_id, display_name, description=description,
                                                        owner=owner, tags=tags, is_internal=is_internal))

    def create_pending_run(self, pipeline_id: str, *, display_name: Optional[str] = None,
                           trigger: TriggerType = TriggerType.SCHEDULE, parameters: Optional[Mapping] = None,
                           metadata: Optional[Mapping] = None, requested_by: Any = None,
                           code_version: Optional[str] = None, actor: Any = None,
                           is_internal: bool = False) -> PipelineRun:
        """Record a planned run as PENDING (e.g. by a scheduler). It runs only when its pipeline claims it."""
        return self._store.create_run(
            self._spec(pipeline_id, display_name, is_internal=is_internal), actor=self._actor(actor, pipeline_id),
            trigger_type=trigger, start=False,
            requested_by=None if requested_by is None else self._redactor.sanitize_actor(requested_by),
            parameters=self._redactor.sanitize_mapping(parameters),
            metadata=self._redactor.sanitize_mapping(metadata), code_version=code_version)

    def start_run(self, pipeline_id: str, *, display_name: Optional[str] = None,
                  trigger: TriggerType = TriggerType.MANUAL, parameters: Optional[Mapping] = None,
                  metadata: Optional[Mapping] = None, requested_by: Any = None, code_version: Optional[str] = None,
                  actor: Any = None, is_internal: bool = False) -> "RunHandle":
        """Create a RUNNING run and return its handle (prefer :meth:`track_run`, which also finishes it)."""
        actor_value = self._actor(actor, pipeline_id)
        start_ns = self._clock.monotonic_ns()
        run = self._store.create_run(
            self._spec(pipeline_id, display_name, is_internal=is_internal), actor=actor_value, trigger_type=trigger,
            start=True, requested_by=None if requested_by is None else self._redactor.sanitize_actor(requested_by),
            parameters=self._redactor.sanitize_mapping(parameters),
            metadata=self._redactor.sanitize_mapping(metadata), code_version=code_version)
        return RunHandle(self, run, actor_value, start_ns)

    def claim_pending_run(self, pipeline_run_id: str, *, pipeline_id: Optional[str] = None,
                          actor: Any = None) -> "RunHandle":
        """PENDING → RUNNING for a run this process will execute (e.g. a requested retry). Atomic: one claimer wins."""
        current = self._store.get_run(pipeline_run_id)
        if pipeline_id is not None and current.pipeline_id != pipeline_id:
            raise ObservabilityValidationError("The pending run belongs to a different pipeline.")
        actor_value = self._actor(actor, current.pipeline_id)
        start_ns = self._clock.monotonic_ns()
        run = self._store.start_run(pipeline_run_id, actor=actor_value)
        return RunHandle(self, run, actor_value, start_ns)

    @contextmanager
    def track_run(self, pipeline_id: str, *, display_name: Optional[str] = None,
                  trigger: TriggerType = TriggerType.MANUAL, parameters: Optional[Mapping] = None,
                  metadata: Optional[Mapping] = None, requested_by: Any = None, code_version: Optional[str] = None,
                  actor: Any = None, attach_to: Optional[str] = None, is_internal: bool = False) -> Iterator["RunHandle"]:
        """Track one run of a pipeline for the duration of the ``with`` block.

        ``attach_to`` claims an existing PENDING attempt (such as a requested retry) instead of creating a run.
        """
        if attach_to is not None:
            handle = self.claim_pending_run(attach_to, pipeline_id=pipeline_id, actor=actor)
        else:
            handle = self.start_run(pipeline_id, display_name=display_name, trigger=trigger, parameters=parameters,
                                    metadata=metadata, requested_by=requested_by, code_version=code_version,
                                    actor=actor, is_internal=is_internal)
        try:
            yield handle
        except BaseException as exc:
            if not handle.closed:
                try:
                    if isinstance(exc, Exception):
                        handle.fail(exc)
                    else:  # KeyboardInterrupt, SystemExit, GeneratorExit, ...: the owner stopped the run
                        handle.cancel(f"interrupted ({type(exc).__name__})")
                except Exception as tracking_error:  # noqa: BLE001 - never mask the pipeline's own exception
                    _note_tracking_error(exc, tracking_error)
            raise
        else:
            if not handle.closed:
                handle.complete()

    def request_retry(self, pipeline_run_id: str, *, requested_by: Any, reason: Any) -> PipelineRun:
        """Record a retry request: a new PENDING attempt linked to the latest FAILED/CANCELLED one.

        Nothing is executed, scheduled or re-run. The owning pipeline must claim the attempt itself.
        """
        if not str(reason or "").strip():
            raise ObservabilityValidationError("A retry request needs a reason.")
        return self._store.request_retry(pipeline_run_id, requested_by=self._redactor.sanitize_actor(requested_by),
                                         reason=self._redactor.sanitize_text(reason, max_chars=MAX_REASON_CHARS))

    def cancel_pending_run(self, pipeline_run_id: str, *, actor: Any, reason: Any = None) -> PipelineRun:
        """Cancel a PENDING attempt (e.g. an abandoned retry request). RUNNING runs are refused atomically."""
        try:
            return self._store.cancel_run(
                pipeline_run_id, actor=self._redactor.sanitize_actor(actor),
                reason=None if reason is None else self._redactor.sanitize_text(reason, max_chars=MAX_REASON_CHARS),
                require_status=PipelineRunStatus.PENDING)
        except TerminalStateError:
            raise
        except InvalidTransitionError as exc:
            raise InvalidTransitionError(
                f"{exc} A RUNNING pipeline run can only be cancelled by the process that owns it.") from None


def _note_tracking_error(original: BaseException, tracking_error: BaseException) -> None:
    """Attach a note to the pipeline's exception and log, without leaking values into logs."""
    logger.warning("Pipeline observability could not record a run outcome (%s).", type(tracking_error).__name__)
    if hasattr(original, "add_note"):
        original.add_note(f"[datasentinel] pipeline observability could not record this failure: "
                          f"{type(tracking_error).__name__}")


class TaskHandle:
    """A running task attempt inside a tracked run."""

    def __init__(self, task: PipelineTaskRun) -> None:
        self._task = task
        self._rows: Tuple[Optional[int], Optional[int], Optional[RowsSource]] = (None, None, None)
        self._skipped = False

    @property
    def pipeline_task_run_id(self) -> str:
        return self._task.pipeline_task_run_id

    @property
    def task_name(self) -> str:
        return self._task.task_name

    @property
    def attempt(self) -> int:
        return self._task.task_attempt

    def record_rows(self, input_rows: Optional[int] = None, output_rows: Optional[int] = None, *,
                    source: RowsSource = RowsSource.REPORTED) -> None:
        """Row counts for this task (persisted when the task finishes)."""
        self._rows = (input_rows, output_rows, RowsSource(source))

    def skip(self) -> None:
        """Finish this task as SKIPPED instead of SUCCESS when the block exits normally."""
        self._skipped = True


class RunHandle:
    """A RUNNING pipeline run owned by this process."""

    def __init__(self, tracker: PipelineTracker, run: PipelineRun, actor, start_ns: int) -> None:
        self._tracker = tracker
        self._store = tracker._store
        self._clock = tracker._clock
        self._run = run
        self._actor = actor
        self._start_ns = start_ns
        self._last_heartbeat_ns = start_ns  # the start itself is the first sign of life
        self._task_failures: List[Tuple[BaseException, PipelineTaskRun]] = []
        self._lock = threading.Lock()
        self._closed = False

    @property
    def pipeline_run_id(self) -> str:
        return self._run.pipeline_run_id

    @property
    def attempt_number(self) -> int:
        return self._run.attempt_number

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise TerminalStateError("This tracked pipeline run has already finished.")

    def heartbeat(self, *, force: bool = False) -> bool:
        """Persist a heartbeat unless one was written within the minimum interval. Returns whether it wrote."""
        self._require_open()
        now = self._clock.monotonic_ns()
        if not force and now - self._last_heartbeat_ns < self._tracker._heartbeat_interval_ns:
            return False
        self._store.heartbeat(self.pipeline_run_id)
        self._last_heartbeat_ns = now
        return True

    def record_rows(self, input_rows: Optional[int] = None, output_rows: Optional[int] = None, *,
                    source: RowsSource = RowsSource.REPORTED) -> PipelineRun:
        """Record run-level row counts (facts, labelled measured or reported)."""
        self._require_open()
        self._run = self._store.record_rows(self.pipeline_run_id, input_rows=input_rows, output_rows=output_rows,
                                            rows_source=source, actor=self._actor)
        return self._run

    @contextmanager
    def task(self, task_name: str) -> Iterator[TaskHandle]:
        """Track one task attempt. A raised exception marks it FAILED and propagates unchanged."""
        self._require_open()
        task = self._store.start_task(self.pipeline_run_id, task_name, actor=self._actor)
        handle, start_ns = TaskHandle(task), self._clock.monotonic_ns()
        try:
            yield handle
        except BaseException as exc:
            status = TaskStatus.FAILED if isinstance(exc, Exception) else TaskStatus.CANCELLED
            try:
                # A failed attempt's details are written atomically with the FAILED status (durable across crashes).
                record = self._tracker.build_failure(exc, primary=False, task=task) \
                    if status is TaskStatus.FAILED else None
                self._finish_task(handle, status, start_ns, failure=record)
                if status is TaskStatus.FAILED:
                    with self._lock:
                        self._task_failures.append((exc, task))
            except Exception as tracking_error:  # noqa: BLE001 - never mask the task's own exception
                _note_tracking_error(exc, tracking_error)
            raise
        else:
            self._finish_task(handle, TaskStatus.SKIPPED if handle._skipped else TaskStatus.SUCCESS, start_ns)
        self.heartbeat()

    def _finish_task(self, handle: TaskHandle, status: TaskStatus, start_ns: int,
                     failure: Optional[FailureRecord] = None) -> None:
        input_rows, output_rows, source = handle._rows
        has_rows = input_rows is not None or output_rows is not None
        self._store.finish_task(handle.pipeline_task_run_id, status, actor=self._actor, input_rows=input_rows,
                                output_rows=output_rows, rows_source=source if has_rows else None,
                                monotonic_elapsed_ms=_elapsed_ms(self._clock, start_ns), failure=failure)

    def complete(self) -> PipelineRun:
        """RUNNING → SUCCESS. Failed task attempts were already recorded as non-primary failures."""
        self._require_open()
        self._run = self._store.complete_run(self.pipeline_run_id, actor=self._actor,
                                             monotonic_elapsed_ms=_elapsed_ms(self._clock, self._start_ns))
        self._closed = True
        return self._run

    def fail(self, exc: Optional[BaseException] = None, *, category: Optional[FailureCategory] = None,
             message: Optional[str] = None) -> PipelineRun:
        """RUNNING → FAILED with exactly one primary failure (``exc``, or a reported category/message)."""
        self._require_open()
        if exc is None and category is None:
            raise ObservabilityValidationError("fail() needs the exception or a reported failure category.")
        primary_task = None
        if exc is not None:  # a task's exception that ended the run: the primary failure links to that task
            links = {id(link) for link in exception_chain(exc)}
            with self._lock:
                matches = [t for e, t in self._task_failures if id(e) in links]
            if matches:
                primary_task = matches[-1]
        primary = self._tracker.build_failure(exc, primary=True, task=primary_task, reported_category=category,
                                              message=message)
        self._run = self._store.fail_run(self.pipeline_run_id, [primary],
                                         actor=self._actor,
                                         monotonic_elapsed_ms=_elapsed_ms(self._clock, self._start_ns))
        self._closed = True
        return self._run

    def cancel(self, reason: Any = None) -> PipelineRun:
        """Cancel this RUNNING run (only the owning process may do this)."""
        self._require_open()
        self._run = self._store.cancel_run(
            self.pipeline_run_id, actor=self._actor,
            reason=None if reason is None else self._tracker._redactor.sanitize_text(reason, max_chars=MAX_REASON_CHARS),
            monotonic_elapsed_ms=_elapsed_ms(self._clock, self._start_ns))
        self._closed = True
        return self._run


__all__ = ["PipelineTracker", "RunHandle", "TaskHandle"]
