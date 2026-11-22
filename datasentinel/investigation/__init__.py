"""Deterministic investigation support: run persistence, workspaces, drill-down queries, privacy.

Nothing in this package calls an LLM. The agent layer (``datasentinel_agents``) consumes
these functions as read-only tools.
"""

from datasentinel.investigation.drilldown import (
    ISSUE_TYPES,
    DrilldownError,
    issue_mask,
    issue_rate_by_segment,
    issue_trend,
    missingness_cooccurrence,
    sample_issue_rows,
)
from datasentinel.investigation.privacy import PRIVACY_MODES, PrivacyGuard, shape_signature
from datasentinel.investigation.workspace import RunManifest, RunStore, Workspace, WorkspaceManager

__all__ = [
    "ISSUE_TYPES",
    "PRIVACY_MODES",
    "DrilldownError",
    "PrivacyGuard",
    "RunManifest",
    "RunStore",
    "Workspace",
    "WorkspaceManager",
    "issue_mask",
    "issue_rate_by_segment",
    "issue_trend",
    "missingness_cooccurrence",
    "sample_issue_rows",
    "shape_signature",
]
