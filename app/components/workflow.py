"""Derive investigation stage statuses from the checkpointed LangGraph state (pure, testable).

Only facts present in the state are used: agent statuses recorded by the graph, which report
objects exist, the run status and the next scheduled nodes. Nothing is inferred from timing and
no progress is fabricated. Routing is a conditional edge (not a node), so it is shown as part of
"Prepare & route".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

StageStatus = Literal["done", "skipped", "failed", "degraded", "waiting", "interrupted", "not_reached"]


@dataclass(frozen=True)
class Stage:
    """One workflow stage as displayed."""

    key: str
    label: str
    status: StageStatus
    detail: str = ""


_AGENT_STAGES = {
    "investigate": ("data_investigator", "Data investigation"),
    "schema": ("schema_analyst", "Schema analysis"),
    "root_cause": ("root_cause_analyst", "Root-cause analysis"),
    "remediation": ("remediation_agent", "Remediation proposal"),
}
_NODE_FOR_STAGE = {
    "prepare": "prepare_context", "investigate": "data_investigator", "schema": "schema_analyst",
    "grounding": "grounding_check", "root_cause": "root_cause_analyst", "remediation": "remediation_agent",
    "validation": "validate_fixes", "approval": "human_approval", "execution": "execute_fixes",
    "verification": "verify", "report": "report",
}


def derive_stages(values: dict[str, Any], run_status: str, next_nodes: tuple[str, ...] = ()) -> list[Stage]:
    """Stage list for a run snapshot (``values`` = checkpointed state)."""
    finished = values.get("final_report") is not None
    stopped_at = set(next_nodes) if run_status == "incomplete" else set()
    agents = values.get("agent_status") or {}

    def fallback(stage_key: str) -> StageStatus:
        if _NODE_FOR_STAGE[stage_key] in stopped_at:
            return "interrupted"
        return "skipped" if finished else "not_reached"

    stages: list[Stage] = [Stage("prepare", "Prepare & route",
                                 "done" if values.get("digest") is not None else fallback("prepare"))]

    def agent_stage(key: str) -> Stage:
        node, label = _AGENT_STAGES[key]
        status = getattr(agents.get(node), "status", "pending")
        mapping = {"succeeded": "done", "degraded": "degraded", "failed": "failed", "skipped": "skipped"}
        message = getattr(agents.get(node), "message", "") or ""
        return Stage(key, label, mapping.get(status, fallback(key)), message)  # type: ignore[arg-type]

    stages += [agent_stage("investigate"), agent_stage("schema")]
    grounded = any(getattr(agents.get(n), "status", "pending") not in ("pending", "skipped")
                   for n in ("root_cause_analyst",))
    stages.append(Stage("grounding", "Evidence grounding", "done" if grounded else fallback("grounding")))
    stages += [agent_stage("root_cause"), agent_stage("remediation")]

    validations = values.get("fix_validations")
    if validations is not None and values.get("remediation_plan") is not None:
        approvable = sum(1 for v in validations if v.approvable)
        stages.append(Stage("validation", "Fix validation", "done", f"{approvable} of {len(validations)} approvable"))
    else:
        stages.append(Stage("validation", "Fix validation", fallback("validation")))

    approvals = values.get("approvals")
    if run_status == "awaiting_approval":
        stages.append(Stage("approval", "Human approval", "waiting", "Waiting for a reviewer"))
    elif approvals is not None:
        approved = sum(1 for a in approvals if a.decision == "approved")
        stages.append(Stage("approval", "Human approval", "done", f"{approved} approved, {len(approvals) - approved} rejected"))
    else:
        stages.append(Stage("approval", "Human approval", fallback("approval")))

    execution = values.get("execution")
    if execution is not None:
        status: StageStatus = "done" if execution.artifact_name else "failed"
        stages.append(Stage("execution", "Execution", status,
                            "New artefact written" if execution.artifact_name else "No fix executed successfully"))
    else:
        stages.append(Stage("execution", "Execution", fallback("execution")))

    stages.append(Stage("verification", "Verification",
                        "done" if values.get("verification") is not None else fallback("verification")))
    stages.append(Stage("report", "Final report", "done" if finished else fallback("report")))
    return stages
