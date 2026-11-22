"""Deterministic control nodes: grounding, fix validation, approval, execution, verification, report."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.types import interrupt

from datasentinel.quality.models import CheckStatus
from datasentinel.remediation import ApprovedFix, SandboxError, check_pyspark, execute_approved_fixes, sql_sha256, validate_sql
from datasentinel.remediation.audit import AuditLog
from datasentinel_agents.grounding import ground_investigator_report, ground_schema_report
from datasentinel_agents.models import (
    ApprovalDecision,
    ExecutionResult,
    FixExecution,
    FixValidation,
    InvestigationReport,
    ProposedFix,
    VerificationResult,
)
from datasentinel_agents.nodes.common import analyze_frame, now
from datasentinel_agents.services import Services
from datasentinel_agents.state import InvestigationState

logger = logging.getLogger(__name__)
Node = Callable[[InvestigationState], dict[str, Any]]


# ---------------------------------------------------------------------- grounding


def make_grounding_node(services: Services) -> Node:
    """Ground investigator and schema reports against the evidence ledger."""

    def grounding_check(state: InvestigationState) -> dict[str, Any]:
        ledger = {e.evidence_id: e for e in state.get("evidence") or []}
        update: dict[str, Any] = {"errors": []}
        investigator = state.get("investigator_report")
        if investigator is not None:
            update["errors"] += ground_investigator_report(investigator, ledger)
            update["investigator_report"] = investigator
        schema_report = state.get("schema_report")
        if schema_report is not None:
            diff = services.workspaces.get(state["run_id"]).schema_diff
            update["errors"] += ground_schema_report(schema_report, ledger, diff)
            update["schema_report"] = schema_report
        logger.info("Grounding produced %d note(s)", len(update["errors"]))
        return update

    return grounding_check


# ---------------------------------------------------------------------- fix validation


def make_validate_fixes_node(services: Services) -> Node:
    """Validate every proposed fix: SQL policy, PySpark static check, sandbox dry run, before/after quality."""

    def validate_fixes(state: InvestigationState) -> dict[str, Any]:
        plan = state.get("remediation_plan")
        if plan is None or not plan.fixes:
            return {"fix_validations": []}
        workspace = services.workspaces.get(state["run_id"])
        raw = workspace.result.raw_dataframe
        sample = raw.head(services.agent_settings.dry_run_rows).reset_index(drop=True)
        before = analyze_frame(workspace, sample, "dry_run_before").quality
        validations = []
        for fix in plan.fixes:
            sql_check, py_check = validate_sql(fix.sql), check_pyspark(fix.pyspark)
            record = FixValidation(fix_id=fix.fix_id, sql_sha256=sql_check.sha256, sql_ok=sql_check.ok,
                                   sql_violations=sql_check.violations, pyspark_ok=py_check.ok,
                                   pyspark_violations=py_check.violations, rows_before=len(sample),
                                   score_before=before.overall_score,
                                   failed_checks_before=before.summary()["by_status"]["failed"])
            if sql_check.ok:
                try:
                    output = services.sandbox.run(fix.sql, sample)
                    record.dry_run_ok = True
                    record.rows_after = len(output)
                    record.columns_preserved = set(output.columns) == set(raw.columns)
                    if record.columns_preserved and len(output) > 0:
                        after = analyze_frame(workspace, output[list(raw.columns)], "dry_run_after").quality
                        record.score_after = after.overall_score
                        record.failed_checks_after = after.summary()["by_status"]["failed"]
                except SandboxError as exc:
                    record.dry_run_error = str(exc)[:500]
            validations.append(record)
        logger.info("Validated %d fix(es); %d approvable", len(validations), sum(v.approvable for v in validations))
        return {"fix_validations": validations}

    return validate_fixes


# ---------------------------------------------------------------------- human approval


def approval_payload(state: InvestigationState) -> dict[str, Any]:
    """The interrupt payload shown to reviewers: only fixes that passed every deterministic gate."""
    validations = {v.fix_id: v for v in state.get("fix_validations") or [] if v.approvable}
    plan = state.get("remediation_plan")
    fixes = [f for f in (plan.fixes if plan else []) if f.fix_id in validations]
    return {
        "type": "fix_approval",
        "run_id": state["run_id"],
        "instructions": "Approve or reject each fix. Approval is bound to the SQL SHA-256 shown.",
        "fixes": [{**f.model_dump(), "sql_sha256": validations[f.fix_id].sql_sha256,
                   "validation": validations[f.fix_id].model_dump()} for f in fixes],
    }


def parse_decisions(response: Any, state: InvestigationState) -> list[ApprovalDecision]:
    """Turn a resume payload into decisions. Anything malformed or mismatched becomes a rejection."""
    validations = {v.fix_id: v for v in state.get("fix_validations") or [] if v.approvable}
    plan = state.get("remediation_plan")
    fixes: dict[str, ProposedFix] = {f.fix_id: f for f in (plan.fixes if plan else []) if f.fix_id in validations}
    raw_items = response.get("decisions") if isinstance(response, dict) else None
    items = {str(i.get("fix_id")): i for i in raw_items if isinstance(i, dict)} if isinstance(raw_items, list) else {}
    decisions = []
    for fix_id, fix in fixes.items():
        expected = validations[fix_id].sql_sha256
        item = items.get(fix_id)
        reviewer = str(item.get("reviewer") or "").strip()[:100] if item else ""
        base = {"fix_id": fix_id, "sql_sha256": expected, "decided_at": now(), "reviewer": reviewer or "unknown",
                "comment": (str(item.get("comment"))[:1000] if item and item.get("comment") else None)}
        if item is None:
            decisions.append(ApprovalDecision(**base, decision="rejected", system_note="No decision provided"))
        elif not reviewer:
            decisions.append(ApprovalDecision(**base, decision="rejected", system_note="Reviewer identity is required"))
        elif item.get("sql_sha256") != expected or sql_sha256(fix.sql) != expected:
            decisions.append(ApprovalDecision(**base, decision="rejected",
                                              system_note="SQL hash mismatch; approval is bound to the exact SQL"))
        elif item.get("decision") == "approved":
            decisions.append(ApprovalDecision(**base, decision="approved"))
        else:
            decisions.append(ApprovalDecision(**base, decision="rejected"))
    return decisions


def make_approval_node(services: Services) -> Node:
    """Pause for human review via ``interrupt()``; resume with ``Command(resume={"decisions": [...]})``."""

    def human_approval(state: InvestigationState) -> dict[str, Any]:
        response = interrupt(approval_payload(state))
        decisions = parse_decisions(response, state)
        audit = services.audit(state["run_id"])
        for decision in decisions:
            audit.record("approval_decision", fix_id=decision.fix_id, decision=decision.decision,
                         reviewer=decision.reviewer, sql_sha256=decision.sql_sha256, comment=decision.comment,
                         system_note=decision.system_note)
        logger.info("Approval decisions: %s", {d.fix_id: d.decision for d in decisions})
        return {"approvals": decisions}

    return human_approval


# ---------------------------------------------------------------------- execution & verification


def make_execute_node(services: Services) -> Node:
    """Execute approved fixes only. Hash and policy are re-checked inside the executor."""

    def execute_fixes(state: InvestigationState) -> dict[str, Any]:
        run_id = state["run_id"]
        workspace = services.workspaces.get(run_id)
        plan = state["remediation_plan"]
        approved = {a.fix_id: a for a in state.get("approvals") or [] if a.decision == "approved"}
        by_id = {f.fix_id: f for f in plan.fixes}
        ordered = [fid for fid in plan.execution_order if fid in approved and fid in by_id]
        fixes = [ApprovedFix(fid, by_id[fid].sql, approved[fid].sql_sha256, approved[fid].reviewer) for fid in ordered]
        raw = workspace.result.raw_dataframe
        audit = AuditLog(services.store.audit_path(run_id))
        audit.record("execution_started", fix_ids=[f.fix_id for f in fixes])
        outcome = execute_approved_fixes(raw, fixes, services.sandbox, audit)
        artifact = None
        if outcome.output is not None:
            artifact = services.store.next_artifact_name(run_id)
            services.store.save_artifact(run_id, artifact, outcome.output)
            audit.record("artifact_saved", artifact=artifact, rows=len(outcome.output), fix_ids=outcome.executed_ids)
        result = ExecutionResult(
            records=[FixExecution(**r.__dict__) for r in outcome.records],
            artifact_name=artifact,
            rows_before=len(raw),
            rows_after=None if outcome.output is None else len(outcome.output),
        )
        return {"execution": result}

    return execute_fixes


def make_verify_node(services: Services) -> Node:
    """Re-run the deterministic engine on the remediated dataset and compare."""

    def verify(state: InvestigationState) -> dict[str, Any]:
        execution = state.get("execution")
        if execution is None or execution.artifact_name is None:
            return {}
        workspace = services.workspaces.get(state["run_id"])
        df = services.store.load_artifact(state["run_id"], execution.artifact_name)
        before = workspace.result.quality
        after = analyze_frame(workspace, df, execution.artifact_name).quality

        def failed(report) -> set[str]:
            return {f"{r.rule_id}:{r.column or '(dataset)'}" for r in report.results if r.status == CheckStatus.FAILED}

        after_dims = {d.dimension.value: d.score for d in after.dimension_scores}
        deltas = {d.dimension.value: (None if d.score is None or after_dims.get(d.dimension.value) is None
                                      else round(after_dims[d.dimension.value] - d.score, 2))
                  for d in before.dimension_scores}
        return {"verification": VerificationResult(
            score_before=before.overall_score, score_after=after.overall_score,
            grade_before=before.grade, grade_after=after.grade,
            failed_checks_before=len(failed(before)), failed_checks_after=len(failed(after)),
            dimension_deltas=deltas,
            resolved_checks=sorted(failed(before) - failed(after)),
            new_failures=sorted(failed(after) - failed(before)),
        )}

    return verify


# ---------------------------------------------------------------------- report


def _outcome(state: InvestigationState) -> str:
    if state.get("verification") is not None:
        return "remediated"
    execution = state.get("execution")
    if execution is not None and execution.artifact_name is None:
        return "execution_failed"  # approved fixes were all blocked or failed; nothing was written
    if state.get("approvals") is not None and state.get("execution") is None:
        return "fixes_rejected"
    if state.get("remediation_plan") is not None:
        return "no_valid_fixes" if not any(v.approvable for v in state.get("fix_validations") or []) else "fixes_proposed"
    if state.get("root_cause_report") is not None or state.get("investigator_report") is not None:
        return "investigated"
    return "no_issues"


def make_report_node(services: Services) -> Node:
    """Assemble the final report. Quality metrics are copied from the engine."""

    def report(state: InvestigationState) -> dict[str, Any]:
        workspace = services.workspaces.get(state["run_id"])
        quality = workspace.result.quality
        investigator = state.get("investigator_report")
        # Agents still "pending" at the end were never routed to; say so explicitly.
        statuses = {name: (status.model_copy(update={"status": "skipped", "message": "Not required for this run"})
                           if status.status == "pending" else status)
                    for name, status in (state.get("agent_status") or {}).items()}
        incomplete = sorted(name for name, status in statuses.items() if status.status in ("failed", "degraded"))
        return {"agent_status": statuses, "final_report": InvestigationReport(
            run_id=state["run_id"],
            dataset_name=workspace.manifest.filename,
            privacy_mode=workspace.guard.mode,
            generated_at=now(),
            quality_score=quality.overall_score,
            grade=quality.grade,
            findings=investigator.findings if investigator else [],
            schema_analysis=state.get("schema_report"),
            root_cause=state.get("root_cause_report"),
            remediation=state.get("remediation_plan"),
            validations=state.get("fix_validations") or [],
            approvals=state.get("approvals") or [],
            execution=state.get("execution"),
            verification=state.get("verification"),
            agent_status=statuses,
            incomplete_agents=incomplete,
            errors=state.get("errors") or [],
            evidence_count=len(state.get("evidence") or []),
            outcome=_outcome(state),
        )}

    return report
