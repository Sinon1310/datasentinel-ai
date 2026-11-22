"""The four LLM agent nodes.

Each node builds a privacy-filtered task from shared state, runs :class:`AgentRunner` with
that agent's tool allow-list, grounds the result where applicable, and writes back only
validated, structured output plus evidence, status, errors and usage.

Error policy: transient provider errors are re-raised so the node's ``RetryPolicy`` retries
it (and, if retries are exhausted, the run stops at a resumable checkpoint). Any other
failure degrades the agent and the graph continues with partial results.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel

from datasentinel_agents.evidence import AGENT_PREFIXES, EvidenceRecorder, next_index
from datasentinel_agents.grounding import ground_remediation_plan, ground_root_cause_report
from datasentinel_agents.llm import is_transient_error
from datasentinel_agents.models import (
    AgentRunStatus,
    Finding,
    InvestigatorReport,
    RemediationPlan,
    RootCauseReport,
    SchemaAnalysisReport,
)
from datasentinel_agents.nodes.common import agent_error, confirmed_findings
from datasentinel_agents.prompts import load_prompt
from datasentinel_agents.runtime import AgentOutcome, AgentRunner
from datasentinel_agents.services import Services
from datasentinel_agents.state import InvestigationState
from datasentinel_agents.tools import ToolContext, build_tools

logger = logging.getLogger(__name__)
Node = Callable[[InvestigationState], dict[str, Any]]


def _run(services: Services, state: InvestigationState, agent: str, output_model: type[BaseModel],
         task: dict[str, Any], findings: list[Finding] | None = None) -> AgentOutcome:
    workspace = services.workspaces.get(state["run_id"])
    existing = state.get("evidence") or []
    recorder = EvidenceRecorder(agent, start=next_index(existing, AGENT_PREFIXES[agent]))
    ctx = ToolContext(workspace=workspace, recorder=recorder, settings=services.agent_settings,
                      sandbox=services.sandbox, ledger={e.evidence_id: e for e in existing}, findings=findings or [])
    runner = AgentRunner(services.llm_factory(agent), services.agent_settings)
    return runner.run(agent, load_prompt(agent), task, build_tools(agent, ctx), output_model, recorder.records)


def _base_update(outcome: AgentOutcome) -> dict[str, Any]:
    return {
        "evidence": outcome.evidence,
        "agent_status": {outcome.agent: outcome.status},
        "errors": outcome.errors,
        "usage": {outcome.agent: outcome.usage},
    }


def _guarded(agent: str, body: Callable[[InvestigationState], dict[str, Any]]) -> Node:
    def node(state: InvestigationState) -> dict[str, Any]:
        try:
            return body(state)
        except Exception as exc:
            if is_transient_error(exc):
                logger.warning("%s hit a transient LLM error; retrying via RetryPolicy: %s", agent, exc)
                raise
            logger.exception("%s failed", agent)
            return {
                "agent_status": {agent: AgentRunStatus(status="failed", message=f"{type(exc).__name__}: {str(exc)[:300]}")},
                "errors": [agent_error(agent, "internal", f"{type(exc).__name__}: {exc}")],
            }

    node.__name__ = agent
    return node


# ---------------------------------------------------------------------- data investigator


def make_investigator_node(services: Services) -> Node:
    """Factory for the Data Investigator node."""

    def body(state: InvestigationState) -> dict[str, Any]:
        digest = state["digest"]
        previous = state.get("investigator_report")
        requests = state.get("followup_requests") or []
        followup = bool(requests and previous)
        task: dict[str, Any] = {
            "dataset": digest.model_dump(exclude={"schema_columns"}),
            "columns": [c.model_dump() for c in digest.schema_columns],
            "instructions": "Investigate the failed checks. Cite the seeded E-Q evidence ids.",
        }
        if followup:
            task["followup_requests"] = [r.model_dump() for r in requests]
            task["existing_findings"] = [f.model_dump(include={"finding_id", "title", "columns", "status"}) for f in previous.findings]
            task["instructions"] = "Answer the follow-up requests with targeted tool calls; add only NEW findings."
        outcome = _run(services, state, "data_investigator", InvestigatorReport, task,
                       findings=previous.findings if previous else None)
        update = _base_update(outcome)
        if outcome.output is not None:
            update["investigator_report"] = _merge(previous, outcome.output) if followup else outcome.output
        if followup:
            update["followup_rounds"] = state.get("followup_rounds", 0) + 1
            update["followup_requests"] = []
        return update

    return _guarded("data_investigator", body)


def _merge(previous: InvestigatorReport, new: InvestigatorReport) -> InvestigatorReport:
    existing = {f.finding_id for f in previous.findings}
    added = []
    for finding in new.findings:
        while finding.finding_id in existing:
            finding.finding_id = f"{finding.finding_id}b"
        existing.add(finding.finding_id)
        added.append(finding)
    return InvestigatorReport(
        summary=f"{previous.summary}\n\nFollow-up: {new.summary}".strip(),
        findings=[*previous.findings, *added],
        columns_inspected=list(dict.fromkeys([*previous.columns_inspected, *new.columns_inspected])),
        open_questions=new.open_questions,
    )


# ---------------------------------------------------------------------- schema analyst


def make_schema_analyst_node(services: Services) -> Node:
    """Factory for the Schema Analyst node."""

    def body(state: InvestigationState) -> dict[str, Any]:
        digest = state["digest"]
        changes = [{"evidence_id": e.evidence_id, "summary": e.summary, **{k: e.data.get(k) for k in
                    ("change_type", "column", "before", "after", "compatibility", "reason")}}
                   for e in state.get("evidence") or [] if e.source == "drift_engine"]
        task = {
            "dataset": {"name": digest.dataset_name, "rows": digest.rows, "columns": digest.columns},
            "drift_summary": digest.drift_summary,
            "schema_changes": changes,
            "instructions": "Assess every deterministic schema change. Cite its E-S evidence id.",
        }
        outcome = _run(services, state, "schema_analyst", SchemaAnalysisReport, task)
        update = _base_update(outcome)
        if outcome.output is not None:
            update["schema_report"] = outcome.output
        return update

    return _guarded("schema_analyst", body)


# ---------------------------------------------------------------------- root cause analyst


def make_root_cause_node(services: Services) -> Node:
    """Factory for the Root Cause Analyst node."""

    def body(state: InvestigationState) -> dict[str, Any]:
        digest = state["digest"]
        investigator = state.get("investigator_report")
        schema_report = state.get("schema_report")
        findings = investigator.findings if investigator else []
        task = {
            "dataset": {"name": digest.dataset_name, "rows": digest.rows, "overall_score": digest.overall_score,
                        "grade": digest.grade, "dimension_scores": digest.dimension_scores},
            "findings": [f.model_dump() for f in findings],
            "schema_analysis": schema_report.model_dump() if schema_report else None,
            "followup_allowed": state.get("followup_rounds", 0) < services.agent_settings.max_followup_rounds,
            "instructions": "Separate confirmed facts from hypotheses; assess downstream impact.",
        }
        outcome = _run(services, state, "root_cause_analyst", RootCauseReport, task, findings=findings)
        update = _base_update(outcome)
        if outcome.output is not None:
            report: RootCauseReport = outcome.output
            ledger = {e.evidence_id: e for e in [*(state.get("evidence") or []), *outcome.evidence]}
            update["errors"] = [*outcome.errors, *ground_root_cause_report(report, ledger, {f.finding_id for f in findings})]
            update["root_cause_report"] = report
            allowed = state.get("followup_rounds", 0) < services.agent_settings.max_followup_rounds
            update["followup_requests"] = report.needs_more_evidence if allowed and investigator else []
        return update

    return _guarded("root_cause_analyst", body)


# ---------------------------------------------------------------------- remediation agent


def make_remediation_node(services: Services) -> Node:
    """Factory for the Remediation Agent node."""

    def body(state: InvestigationState) -> dict[str, Any]:
        findings = confirmed_findings(state)
        root = state.get("root_cause_report")
        previous_failures = [
            {"fix_id": v.fix_id, "sql_violations": v.sql_violations, "pyspark_violations": v.pyspark_violations,
             "dry_run_error": v.dry_run_error, "columns_preserved": v.columns_preserved}
            for v in state.get("fix_validations") or [] if not v.approvable
        ]
        task = {
            "confirmed_findings": [f.model_dump(exclude={"grounding_notes"}) for f in findings],
            "root_causes": [c.model_dump() for c in root.root_causes] if root else [],
            "columns": [c.model_dump() for c in state["digest"].schema_columns],
            "max_fixes": services.agent_settings.max_fixes,
            "previous_attempt_violations": previous_failures,
            "instructions": "Propose safe, reviewable fixes. Each SQL must return all original columns.",
        }
        outcome = _run(services, state, "remediation_agent", RemediationPlan, task, findings=findings)
        update = _base_update(outcome)
        update["remediation_rounds"] = state.get("remediation_rounds", 0) + 1
        if outcome.output is not None:
            plan: RemediationPlan = outcome.output
            update["errors"] = [*outcome.errors, *ground_remediation_plan(
                plan, {f.finding_id for f in findings}, services.agent_settings.max_fixes)]
            update["remediation_plan"] = plan
        return update

    return _guarded("remediation_agent", body)
