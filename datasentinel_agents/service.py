"""Investigation service: the single API used by the UI (and later FastAPI).

Lifecycle::

    manifest = service.create_run(bytes, filename, baseline_text=..., privacy_mode=...)
    snap = service.start(manifest.run_id)          # runs until approval interrupt or completion
    snap = service.resume(run_id, decisions)       # after human review
    snap = service.retry(run_id)                   # continue after a transient failure

State is checkpointed in SQLite after every node (``thread_id = run_id``), and datasets are
persisted by the :class:`RunStore`, so any run can be resumed after an application restart.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from datasentinel.investigation.workspace import RunManifest, Workspace
from datasentinel.schema.drift import load_baseline
from datasentinel_agents.graph import build_graph, make_serializer
from datasentinel_agents.services import Services

logger = logging.getLogger(__name__)

RunStatus = Literal["not_started", "awaiting_approval", "completed", "failed", "incomplete"]
UpdateCallback = Callable[[dict[str, Any]], None]


@dataclass
class RunSnapshot:
    """Point-in-time view of a run."""

    run_id: str
    status: RunStatus
    values: dict[str, Any] = field(default_factory=dict)
    pending_approval: dict[str, Any] | None = None
    next_nodes: tuple[str, ...] = ()
    error: str | None = None


class InvestigationService:
    """Create, run, pause, resume and inspect investigations."""

    def __init__(self, services: Services, checkpoint_path: Path | None = None) -> None:
        self.services = services
        path = Path(checkpoint_path or services.settings.data_dir / services.agent_settings.checkpoint_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn, serde=make_serializer())
        self.graph = build_graph(services, self.checkpointer)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ runs

    def create_run(self, data: bytes, filename: str, baseline_text: str | None = None,
                   privacy_mode: str = "metadata_only", iqr_multiplier: float = 1.5,
                   max_null_pct: float = 0.0) -> RunManifest:
        """Persist inputs for a new run.

        The dataset and baseline are validated BEFORE anything is written, so invalid input
        never leaves an orphan run in the history.
        """
        if baseline_text:
            load_baseline(baseline_text)
        self.services.workspaces.loader.load_bytes(data, filename)
        manifest = self.services.store.create_run(data, filename, baseline_text=baseline_text,
                                                  privacy_mode=privacy_mode, iqr_multiplier=iqr_multiplier,
                                                  max_null_pct=max_null_pct)
        self.services.workspaces.get(manifest.run_id)  # fail fast on unreadable data / invalid baseline
        return manifest

    def list_runs(self) -> list[RunManifest]:
        """All persisted runs, newest first."""
        return self.services.store.list_runs()

    def workspace(self, run_id: str) -> Workspace:
        """The (reconstructed if necessary) workspace of a run."""
        return self.services.workspaces.get(run_id)

    # ------------------------------------------------------------------ execution

    def _config(self, run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": self.services.agent_settings.recursion_limit}

    def _stream(self, payload: Any, run_id: str, on_update: UpdateCallback | None) -> RunSnapshot:
        self.services.store.run_dir(run_id)  # validates id
        with self._lock:
            try:
                for chunk in self.graph.stream(payload, self._config(run_id), stream_mode="updates"):
                    if on_update is not None:
                        on_update(chunk)
            except Exception as exc:  # noqa: BLE001 — surfaced as a resumable failure
                logger.exception("Run %s stopped with an error", run_id)
                return self.snapshot(run_id, error=f"{type(exc).__name__}: {exc}")
        return self.snapshot(run_id)

    def start(self, run_id: str, on_update: UpdateCallback | None = None) -> RunSnapshot:
        """Start a run; returns at the approval interrupt or on completion."""
        if self.snapshot(run_id).status != "not_started":
            raise ValueError(f"Run {run_id} has already started.")
        self.services.store.load_manifest(run_id)
        return self._stream({"run_id": run_id}, run_id, on_update)

    def resume(self, run_id: str, decisions: list[dict[str, Any]], on_update: UpdateCallback | None = None) -> RunSnapshot:
        """Resume an awaiting run with human decisions: ``[{fix_id, sql_sha256, decision, reviewer, comment}]``."""
        if self.snapshot(run_id).status != "awaiting_approval":
            raise ValueError(f"Run {run_id} is not awaiting approval.")
        return self._stream(Command(resume={"decisions": decisions}), run_id, on_update)

    def retry(self, run_id: str, on_update: UpdateCallback | None = None) -> RunSnapshot:
        """Continue an incomplete run from its last checkpoint (e.g. after a Groq outage)."""
        if self.snapshot(run_id).status != "incomplete":
            raise ValueError(f"Run {run_id} is not in a retryable state.")
        return self._stream(None, run_id, on_update)

    def snapshot(self, run_id: str, error: str | None = None) -> RunSnapshot:
        """Current state of a run, read from the checkpoint."""
        state = self.graph.get_state(self._config(run_id))
        values = dict(state.values or {})
        interrupts = [i for task in state.tasks for i in getattr(task, "interrupts", ())]
        if not values:
            status: RunStatus = "failed" if error else "not_started"
        elif interrupts:
            status = "awaiting_approval"
        elif not state.next and values.get("final_report") is not None:
            status = "completed"
        else:
            status = "incomplete"
        return RunSnapshot(run_id=run_id, status=status, values=values,
                           pending_approval=interrupts[0].value if interrupts else None,
                           next_nodes=tuple(state.next or ()), error=error)

    def close(self) -> None:
        """Close the checkpoint connection."""
        self._conn.close()
