"""``prepare_context``: deterministic setup of a run (no LLM)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from datasentinel.investigation.privacy import safe_identifier
from datasentinel.investigation.workspace import Workspace
from datasentinel.quality.models import CheckStatus
from datasentinel_agents.evidence import seed_evidence
from datasentinel_agents.models import AgentRunStatus, ColumnDigest, FailedCheckDigest, QualityDigest
from datasentinel_agents.services import Services
from datasentinel_agents.state import InvestigationState

logger = logging.getLogger(__name__)
AGENTS = ("data_investigator", "schema_analyst", "root_cause_analyst", "remediation_agent")


def build_digest(workspace: Workspace, check_index: dict[tuple[str, str | None], str]) -> QualityDigest:
    """Compact, privacy-filtered summary of the deterministic analysis."""
    result = workspace.result
    quality, schema = result.quality, result.schema
    failed = [
        FailedCheckDigest(
            evidence_id=check_index[(r.rule_id, r.column)],
            rule_id=r.rule_id,
            column=safe_identifier(r.column) if r.column else None,
            severity=r.severity.value,
            affected_rows=r.affected_rows,
            affected_pct=round(r.affected_pct, 4),
            message=r.message,
        )
        for r in quality.failed_results
    ]
    diff = workspace.schema_diff
    return QualityDigest(
        dataset_name=safe_identifier(workspace.manifest.filename),
        rows=schema.n_rows,
        columns=schema.n_columns,
        overall_score=quality.overall_score,
        grade=quality.grade,
        dimension_scores={d.dimension.value: d.score for d in quality.dimension_scores},
        failed_checks=failed,
        schema_columns=[ColumnDigest(name=safe_identifier(c.name), semantic_type=c.semantic_type.value,
                                     storage_type=c.storage_type, null_count=c.null_count, unique_count=c.unique_count)
                        for c in schema.columns],
        primary_key=schema.primary_key.label if schema.primary_key else None,
        privacy_mode=workspace.guard.mode,
        has_baseline=workspace.baseline is not None,
        drift_summary=diff.to_dict()["summary"] if diff else None,
    )


def make_prepare_node(services: Services) -> Callable[[InvestigationState], dict[str, Any]]:
    """Factory for the ``prepare_context`` node."""

    def prepare_context(state: InvestigationState) -> dict[str, Any]:
        workspace = services.workspaces.get(state["run_id"])
        evidence, index = seed_evidence(workspace)
        digest = build_digest(workspace, index)
        failed = sum(1 for r in workspace.result.quality.results if r.status == CheckStatus.FAILED)
        logger.info("Run %s prepared: %d failed checks, %d evidence seeded", state["run_id"], failed, len(evidence))
        return {
            "digest": digest,
            "evidence": evidence,
            "followup_rounds": 0,
            "remediation_rounds": 0,
            "followup_requests": [],
            "agent_status": {a: AgentRunStatus(status="pending") for a in AGENTS},
        }

    return prepare_context
