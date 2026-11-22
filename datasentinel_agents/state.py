"""Shared LangGraph state (the blackboard).

Agents never message each other; each reads specific typed fields and writes one
validated report. Reducers make parallel branches (investigator ‖ schema analyst) merge
safely. Agent chat histories are private to each node and never stored here.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from datasentinel_agents.models import (
    AgentError,
    AgentRunStatus,
    ApprovalDecision,
    Evidence,
    ExecutionResult,
    FixValidation,
    FollowUpRequest,
    InvestigationReport,
    InvestigatorReport,
    QualityDigest,
    RemediationPlan,
    RootCauseReport,
    SchemaAnalysisReport,
    TokenUsage,
    VerificationResult,
)


def merge_evidence(left: list[Evidence] | None, right: list[Evidence] | None) -> list[Evidence]:
    """Append-only ledger merge; the first record for an id wins (records are immutable)."""
    merged = list(left or [])
    seen = {e.evidence_id for e in merged}
    for item in right or []:
        if item.evidence_id not in seen:
            merged.append(item)
            seen.add(item.evidence_id)
    return merged


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Shallow dict merge (right wins)."""
    return {**(left or {}), **(right or {})}


class InvestigationState(TypedDict, total=False):
    """State shared by all nodes; persisted in SQLite checkpoints (thread_id = run_id)."""

    # identity / deterministic inputs
    run_id: str
    digest: QualityDigest

    # blackboard
    evidence: Annotated[list[Evidence], merge_evidence]
    investigator_report: InvestigatorReport | None
    schema_report: SchemaAnalysisReport | None
    root_cause_report: RootCauseReport | None
    followup_requests: list[FollowUpRequest]
    remediation_plan: RemediationPlan | None
    fix_validations: list[FixValidation]
    approvals: list[ApprovalDecision]
    execution: ExecutionResult | None
    verification: VerificationResult | None
    final_report: InvestigationReport | None

    # control / observability
    agent_status: Annotated[dict[str, AgentRunStatus], merge_dicts]
    errors: Annotated[list[AgentError], operator.add]
    usage: Annotated[dict[str, TokenUsage], merge_dicts]
    followup_rounds: int
    remediation_rounds: int
