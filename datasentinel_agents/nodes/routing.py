"""Deterministic conditional routing (no LLM decides control flow)."""

from __future__ import annotations

from typing import Callable

from datasentinel_agents.config import AgentSettings
from datasentinel_agents.nodes.common import confirmed_findings
from datasentinel_agents.state import InvestigationState


def route_after_prepare(state: InvestigationState) -> list[str]:
    """Fan out to the investigator (failed checks) and/or schema analyst (drift), or finish."""
    digest = state["digest"]
    targets = []
    if digest.failed_checks:
        targets.append("data_investigator")
    if digest.drift_summary and sum(digest.drift_summary.values()) > 0:
        targets.append("schema_analyst")
    return targets or ["report"]


def make_route_after_root_cause(settings: AgentSettings) -> Callable[[InvestigationState], str]:
    """Follow-up (bounded), remediation when confirmed findings exist, otherwise report."""

    def route_after_root_cause(state: InvestigationState) -> str:
        if state.get("followup_requests") and state.get("followup_rounds", 0) < settings.max_followup_rounds:
            return "data_investigator"
        return "remediation_agent" if confirmed_findings(state) else "report"

    return route_after_root_cause


def make_route_after_validation(settings: AgentSettings) -> Callable[[InvestigationState], str]:
    """Approval when any fix passed every gate; one bounded repair round; otherwise report."""

    def route_after_validation(state: InvestigationState) -> str:
        validations = state.get("fix_validations") or []
        if any(v.approvable for v in validations):
            return "human_approval"
        if validations and state.get("remediation_rounds", 0) <= settings.max_remediation_repairs:
            return "remediation_agent"
        return "report"

    return route_after_validation


def route_after_approval(state: InvestigationState) -> str:
    """Execute only if at least one fix was approved."""
    return "execute_fixes" if any(a.decision == "approved" for a in state.get("approvals") or []) else "report"
