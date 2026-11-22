"""Helpers shared by graph nodes."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from datasentinel.ingestion.models import DatasetMetadata, LoadedDataset
from datasentinel.investigation.workspace import Workspace
from datasentinel.pipeline import AnalysisResult, analyze_dataset
from datasentinel.quality import QualityConfig
from datasentinel_agents.models import AgentError, Finding
from datasentinel_agents.state import InvestigationState


def now() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def agent_error(agent: str, kind: str, message: str) -> AgentError:
    """Build an :class:`AgentError`."""
    return AgentError(agent=agent, kind=kind, message=message[:1000], timestamp=now())  # type: ignore[arg-type]


def analyze_frame(workspace: Workspace, df: pd.DataFrame, name: str) -> AnalysisResult:
    """Run the deterministic engine on a derived DataFrame with the run's own settings."""
    metadata = DatasetMetadata(filename=name, file_type="derived", size_bytes=0, encoding="n/a",
                               n_rows=int(df.shape[0]), n_columns=int(df.shape[1]), load_seconds=0.0)
    manifest = workspace.manifest
    config = QualityConfig(iqr_multiplier=manifest.iqr_multiplier, max_null_pct=manifest.max_null_pct)
    return analyze_dataset(LoadedDataset(df.reset_index(drop=True), metadata), contract=workspace.baseline,
                           quality_config=config)


def confirmed_findings(state: InvestigationState) -> list[Finding]:
    """Investigator findings that survived grounding as confirmed."""
    report = state.get("investigator_report")
    return [f for f in (report.findings if report else []) if f.status == "confirmed"]
