"""Deterministic grounding of agent claims against the evidence ledger (requirement B).

Rules:

1. Cited evidence ids that do not exist are removed (and logged).
2. A claim marked ``confirmed`` without at least one valid supporting evidence id is
   downgraded to ``hypothesis``.
3. Structured numeric claims (``affected_rows``, ``affected_pct``, ``affected_rows_pct``) must
   match a number present in the cited evidence; otherwise the number is removed and
   the claim downgraded.
4. Deterministic facts are never taken from the LLM: drift flags and compatibility
   classes are overwritten from the drift engine; quality scores come from the engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from datasentinel.schema.drift import SchemaDiff
from datasentinel_agents.evidence import numbers_in
from datasentinel_agents.models import (
    AgentError,
    Evidence,
    Finding,
    Hypothesis,
    InvestigatorReport,
    RemediationPlan,
    RootCauseReport,
    SchemaAnalysisReport,
)

ABS_TOLERANCE = 0.05
REL_TOLERANCE = 0.01
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error(agent: str, message: str) -> AgentError:
    return AgentError(agent=agent, kind="grounding", message=message, timestamp=_now())


def number_supported(value: float, evidence: Iterable[Evidence]) -> bool:
    """True if ``value`` matches (within tolerance) a number in any of the evidence payloads."""
    for item in evidence:
        for candidate in numbers_in(item.data):
            if abs(candidate - value) <= max(ABS_TOLERANCE, abs(candidate) * REL_TOLERANCE):
                return True
    return False


def _valid_ids(ids: list[str], index: dict[str, Evidence]) -> tuple[list[str], list[str]]:
    valid = [i for i in dict.fromkeys(ids) if i in index]
    invalid = [i for i in ids if i not in index]
    return valid, invalid


def _add_note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)


def ground_finding(finding: Finding, index: dict[str, Evidence], agent: str) -> list[AgentError]:
    """Ground a single finding in place."""
    errors: list[AgentError] = []
    valid, invalid = _valid_ids(finding.evidence_ids, index)
    if invalid:
        _add_note(finding.grounding_notes, f"Removed unknown evidence ids: {invalid}")
        errors.append(_error(agent, f"{finding.finding_id}: cited unknown evidence {invalid}"))
    finding.evidence_ids = valid
    cited = [index[i] for i in valid]
    if not valid and finding.status == "confirmed":
        finding.status = "hypothesis"
        _add_note(finding.grounding_notes, "Downgraded to hypothesis: no valid supporting evidence")
    for field_name in ("affected_rows", "affected_pct"):
        value = getattr(finding, field_name)
        if value is not None and not number_supported(float(value), cited):
            setattr(finding, field_name, None)
            if finding.status == "confirmed":
                finding.status = "hypothesis"
            _add_note(finding.grounding_notes, f"Removed {field_name}={value}: not found in cited evidence; downgraded")
            errors.append(_error(agent, f"{finding.finding_id}: unsupported {field_name}={value}"))
    # Severity is a validation result: when engine evidence is cited, the engine's (highest) severity wins.
    engine = [item.data.get("severity") for item in cited if item.source == "quality_engine"]
    engine = [s for s in engine if s in _SEVERITY_RANK]
    if engine:
        authoritative = max(engine, key=_SEVERITY_RANK.__getitem__)
        if finding.severity != authoritative:
            _add_note(finding.grounding_notes,
                      f"Severity corrected from '{finding.severity}' to engine severity '{authoritative}'")
            errors.append(_error(agent, f"{finding.finding_id}: severity '{finding.severity}' overridden by engine"))
            finding.severity = authoritative
    return errors


def ground_investigator_report(report: InvestigatorReport, index: dict[str, Evidence]) -> list[AgentError]:
    """Ground every finding in an investigator report (in place)."""
    errors: list[AgentError] = []
    seen: set[str] = set()
    for finding in report.findings:
        if finding.finding_id in seen:
            finding.finding_id = f"{finding.finding_id}-{len(seen) + 1}"
        seen.add(finding.finding_id)
        errors.extend(ground_finding(finding, index, "data_investigator"))
    return errors


def ground_schema_report(
    report: SchemaAnalysisReport, index: dict[str, Evidence], diff: SchemaDiff | None
) -> list[AgentError]:
    """Ground schema assessments and overwrite deterministic facts from the diff (in place)."""
    errors: list[AgentError] = []
    report.drift_detected = bool(diff and diff.drift_detected)
    truth = {(c.change_type, c.column): c.compatibility for c in (diff.changes if diff else [])}
    for assessment in report.assessments:
        valid, invalid = _valid_ids(assessment.evidence_ids, index)
        if invalid:
            _add_note(assessment.grounding_notes, f"Removed unknown evidence ids: {invalid}")
            errors.append(_error("schema_analyst", f"{assessment.column}: cited unknown evidence {invalid}"))
        assessment.evidence_ids = valid
        key = (assessment.change_type, assessment.column)
        if key not in truth:
            assessment.status = "hypothesis"
            _add_note(assessment.grounding_notes, "Not present in the deterministic schema diff; downgraded")
            errors.append(_error("schema_analyst", f"{key} is not in the deterministic diff"))
        else:
            if assessment.compatibility != truth[key]:
                _add_note(assessment.grounding_notes,
                          f"Compatibility corrected from '{assessment.compatibility}' to deterministic '{truth[key]}'")
                assessment.compatibility = truth[key]
            if not valid:
                assessment.status = "hypothesis"
                _add_note(assessment.grounding_notes, "Downgraded to hypothesis: no valid supporting evidence")
    return errors


def _ground_hypothesis(hypothesis: Hypothesis, index: dict[str, Evidence]) -> list[str]:
    valid, invalid = _valid_ids(hypothesis.supporting_evidence_ids, index)
    hypothesis.supporting_evidence_ids = valid
    hypothesis.contradicting_evidence_ids, invalid_contra = _valid_ids(hypothesis.contradicting_evidence_ids, index)
    if invalid or invalid_contra:
        _add_note(hypothesis.grounding_notes, f"Removed unknown evidence ids: {invalid + invalid_contra}")
    if hypothesis.status == "confirmed" and not valid:
        hypothesis.status = "hypothesis"
        _add_note(hypothesis.grounding_notes, "Downgraded to hypothesis: no valid supporting evidence")
    if hypothesis.status == "confirmed" and hypothesis.contradicting_evidence_ids:
        hypothesis.status = "hypothesis"
        _add_note(hypothesis.grounding_notes, "Downgraded to hypothesis: contradicting evidence exists")
    return invalid + invalid_contra


def ground_root_cause_report(
    report: RootCauseReport, index: dict[str, Evidence], finding_ids: set[str]
) -> list[AgentError]:
    """Ground a root-cause report (in place); unverified 'facts' move to ``unverified_claims``."""
    errors: list[AgentError] = []
    confirmed: list[Hypothesis] = []
    report.unverified_claims = []
    for fact in report.confirmed_facts:
        bad = _ground_hypothesis(fact, index)
        if bad:
            errors.append(_error("root_cause_analyst", f"Fact cited unknown evidence {bad}"))
        if fact.status == "confirmed":
            confirmed.append(fact)
        else:
            report.unverified_claims.append(fact)
    report.confirmed_facts = confirmed
    for cause in report.root_causes:
        unknown = [f for f in cause.related_finding_ids if f not in finding_ids]
        if unknown:
            errors.append(_error("root_cause_analyst", f"{cause.cause_id}: unknown finding ids {unknown}"))
        cause.related_finding_ids = [f for f in cause.related_finding_ids if f in finding_ids]
        for hypothesis in cause.hypotheses:
            bad = _ground_hypothesis(hypothesis, index)
            if bad:
                errors.append(_error("root_cause_analyst", f"{cause.cause_id}: unknown evidence {bad}"))
    impact = report.impact
    valid, _ = _valid_ids(impact.evidence_ids, index)
    impact.evidence_ids = valid
    if impact.affected_rows_pct is not None and not number_supported(impact.affected_rows_pct, [index[i] for i in valid]):
        _add_note(impact.grounding_notes, f"Removed affected_rows_pct={impact.affected_rows_pct}: not in cited evidence")
        errors.append(_error("root_cause_analyst", f"Unsupported affected_rows_pct={impact.affected_rows_pct}"))
        impact.affected_rows_pct = None
    return errors


def ground_remediation_plan(plan: RemediationPlan, finding_ids: set[str], max_fixes: int) -> list[AgentError]:
    """Drop fixes that address no known finding, de-duplicate ids, cap the count (in place)."""
    errors: list[AgentError] = []
    kept = []
    seen: set[str] = set()
    for fix in plan.fixes:
        fix.addresses_finding_ids = [f for f in fix.addresses_finding_ids if f in finding_ids]
        if not fix.addresses_finding_ids:
            errors.append(_error("remediation_agent", f"{fix.fix_id} dropped: addresses no confirmed finding"))
            continue
        if fix.fix_id in seen:
            errors.append(_error("remediation_agent", f"Duplicate fix id {fix.fix_id} dropped"))
            continue
        seen.add(fix.fix_id)
        kept.append(fix)
    if len(kept) > max_fixes:
        errors.append(_error("remediation_agent", f"Plan truncated to {max_fixes} fixes"))
        kept = kept[:max_fixes]
    plan.fixes = kept
    plan.execution_order = [f for f in plan.execution_order if f in seen] or [f.fix_id for f in kept]
    for fix in kept:
        if fix.fix_id not in plan.execution_order:
            plan.execution_order.append(fix.fix_id)
    return errors
