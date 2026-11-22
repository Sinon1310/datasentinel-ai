"""Run persistence and workspace reconstruction.

Graph checkpoints hold only small, serialisable state; DataFrames live in a
:class:`Workspace`. To survive process restarts, :class:`RunStore` persists — per
``run_id`` — the original source bytes, their SHA-256, and every setting that affects the
analysis. Because the engine is deterministic, :class:`WorkspaceManager` can rebuild an
identical workspace from those inputs at any time, verifying integrity on the way.

Layout::

    <data_dir>/runs/<run_id>/
        manifest.json        settings + source checksum
        source.<ext>         original upload bytes (never modified)
        artifacts/*.parquet  remediated outputs (new files only)
        audit.jsonl          append-only approval / execution audit trail
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from datasentinel.exceptions import DataSentinelError
from datasentinel.ingestion import DatasetLoader
from datasentinel.investigation.privacy import PRIVACY_MODES, PrivacyGuard
from datasentinel.pipeline import AnalysisResult, analyze_dataset
from datasentinel.quality import QualityConfig
from datasentinel.schema.contract import SchemaContract
from datasentinel.schema.drift import SchemaDiff, compute_schema_diff, load_baseline

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class RunStoreError(DataSentinelError):
    """A run is missing, malformed or fails integrity checks."""


@dataclass
class RunManifest:
    """Everything needed to rebuild a run's workspace deterministically."""

    run_id: str
    created_at: str
    filename: str
    source_sha256: str
    size_bytes: int
    privacy_mode: str = "metadata_only"
    baseline_text: Optional[str] = None
    iqr_multiplier: float = 1.5
    max_null_pct: float = 0.0
    artifacts: List[str] = field(default_factory=list)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class RunStore:
    """Filesystem-backed persistence for investigation runs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        """Directory of a run. Validates the id to prevent path traversal."""
        if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
            raise RunStoreError(f"Invalid run id: {run_id!r}")
        return self.runs_dir / run_id

    def create_run(
        self,
        data: bytes,
        filename: str,
        baseline_text: Optional[str] = None,
        privacy_mode: str = "metadata_only",
        iqr_multiplier: float = 1.5,
        max_null_pct: float = 0.0,
    ) -> RunManifest:
        """Persist a new run's inputs and return its manifest."""
        if privacy_mode not in PRIVACY_MODES:
            raise RunStoreError(f"Unknown privacy mode '{privacy_mode}'.")
        safe_name = Path(filename).name
        extension = Path(safe_name).suffix.lower()
        manifest = RunManifest(
            run_id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            filename=safe_name,
            source_sha256=_sha256(data),
            size_bytes=len(data),
            privacy_mode=privacy_mode,
            baseline_text=baseline_text,
            iqr_multiplier=iqr_multiplier,
            max_null_pct=max_null_pct,
        )
        directory = self.run_dir(manifest.run_id)
        (directory / "artifacts").mkdir(parents=True, exist_ok=False)
        _atomic_write(directory / f"source{extension}", data)
        self._write_manifest(manifest)
        logger.info("Created run %s for '%s' (%d bytes, privacy=%s)", manifest.run_id, safe_name, len(data), privacy_mode)
        return manifest

    def _write_manifest(self, manifest: RunManifest) -> None:
        payload = json.dumps(asdict(manifest), indent=2).encode()
        _atomic_write(self.run_dir(manifest.run_id) / "manifest.json", payload)

    def load_manifest(self, run_id: str) -> RunManifest:
        """Read a run's manifest."""
        path = self.run_dir(run_id) / "manifest.json"
        if not path.is_file():
            raise RunStoreError(f"Run {run_id} not found.")
        try:
            return RunManifest(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RunStoreError(f"Manifest for run {run_id} is corrupt.") from exc

    def list_runs(self) -> List[RunManifest]:
        """All runs, newest first."""
        manifests = []
        for directory in self.runs_dir.iterdir():
            if directory.is_dir() and _RUN_ID.match(directory.name):
                try:
                    manifests.append(self.load_manifest(directory.name))
                except RunStoreError:
                    logger.warning("Skipping unreadable run directory %s", directory.name)
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def source_bytes(self, run_id: str) -> bytes:
        """Original upload bytes, verified against the recorded checksum."""
        manifest = self.load_manifest(run_id)
        path = self.run_dir(run_id) / f"source{Path(manifest.filename).suffix.lower()}"
        if not path.is_file():
            raise RunStoreError(f"Source data for run {run_id} is missing.")
        data = path.read_bytes()
        if _sha256(data) != manifest.source_sha256:
            raise RunStoreError(f"Source data for run {run_id} failed its integrity check (SHA-256 mismatch).")
        return data

    def save_artifact(self, run_id: str, name: str, df: pd.DataFrame) -> Path:
        """Store a derived dataset as a NEW file. Existing artefacts are never overwritten."""
        if not _ARTIFACT_NAME.match(name):
            raise RunStoreError(f"Invalid artifact name: {name!r}")
        with self._lock:
            path = self.run_dir(run_id) / "artifacts" / f"{name}.parquet"
            if path.exists():
                raise RunStoreError(f"Artifact '{name}' already exists for run {run_id}; refusing to overwrite.")
            df.to_parquet(path, index=False)
            manifest = self.load_manifest(run_id)
            manifest.artifacts.append(name)
            self._write_manifest(manifest)
        return path

    def next_artifact_name(self, run_id: str, base: str = "remediated") -> str:
        """First free artefact name, checking BOTH the manifest and the disk.

        A crash between writing the Parquet file and updating the manifest can leave an
        unlisted file behind; it must never block (or be overwritten by) a later attempt.
        """
        if not _ARTIFACT_NAME.match(base):
            raise RunStoreError(f"Invalid artifact name: {base!r}")
        listed = set(self.load_manifest(run_id).artifacts)
        directory = self.run_dir(run_id) / "artifacts"
        for index in range(1, 1000):
            name = base if index == 1 else f"{base}_{index}"
            if name not in listed and not (directory / f"{name}.parquet").exists():
                return name
        raise RunStoreError(f"No free artifact name for '{base}' in run {run_id}.")

    def load_artifact(self, run_id: str, name: str) -> pd.DataFrame:
        """Read a stored artefact."""
        if not _ARTIFACT_NAME.match(name):
            raise RunStoreError(f"Invalid artifact name: {name!r}")
        path = self.run_dir(run_id) / "artifacts" / f"{name}.parquet"
        if not path.is_file():
            raise RunStoreError(f"Artifact '{name}' not found for run {run_id}.")
        return pd.read_parquet(path)

    def audit_path(self, run_id: str) -> Path:
        """Path of the run's append-only audit log."""
        return self.run_dir(run_id) / "audit.jsonl"


@dataclass
class Workspace:
    """In-memory, fully-analysed view of one run."""

    manifest: RunManifest
    result: AnalysisResult
    guard: PrivacyGuard
    baseline: Optional[SchemaContract] = None
    baseline_source: Optional[str] = None
    schema_diff: Optional[SchemaDiff] = None

    @property
    def run_id(self) -> str:
        """The run id."""
        return self.manifest.run_id


class WorkspaceManager:
    """Cache of workspaces with deterministic reconstruction from a :class:`RunStore`."""

    def __init__(self, store: RunStore, loader: Optional[DatasetLoader] = None) -> None:
        self.store = store
        self.loader = loader or DatasetLoader()
        self._cache: Dict[str, Workspace] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> Workspace:
        """Return the workspace for ``run_id``, rebuilding it from disk if needed."""
        with self._lock:
            if run_id not in self._cache:
                self._cache[run_id] = self._build(run_id)
            return self._cache[run_id]

    def evict(self, run_id: str) -> None:
        """Drop a cached workspace (it will be rebuilt on next access)."""
        with self._lock:
            self._cache.pop(run_id, None)

    def _build(self, run_id: str) -> Workspace:
        manifest = self.store.load_manifest(run_id)
        data = self.store.source_bytes(run_id)
        baseline, source = (None, None)
        if manifest.baseline_text:
            baseline, source = load_baseline(manifest.baseline_text)
        dataset = self.loader.load_bytes(data, manifest.filename)
        config = QualityConfig(iqr_multiplier=manifest.iqr_multiplier, max_null_pct=manifest.max_null_pct)
        result = analyze_dataset(dataset, contract=baseline, quality_config=config)
        diff = compute_schema_diff(baseline, result.schema, result.raw_dataframe, source) if baseline else None
        logger.info("Workspace ready for run %s (score=%s)", run_id, result.quality.overall_score)
        return Workspace(
            manifest=manifest,
            result=result,
            guard=PrivacyGuard(manifest.privacy_mode, result.raw_dataframe),
            baseline=baseline,
            baseline_source=source,
            schema_diff=diff,
        )
