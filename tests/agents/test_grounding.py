"""Tests for evidence grounding (requirement B) and model contracts (requirement C)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datasentinel.schema import SchemaContract, SchemaInferrer
from datasentinel.schema.drift import compute_schema_diff
from datasentinel_agents.evidence import EvidenceRecorder, next_index, numbers_in
from datasentinel_agents.graph import make_serializer
from datasentinel_agents.grounding import (
    ground_investigator_report,
    ground_remediation_plan,
    ground_root_cause_report,
    ground_schema_report,
    number_supported,
)
from datasentinel_agents.models import (
    SUBMISSION_CONTEXT,
    Evidence,
    Finding,
    Hypothesis,
    ImpactAssessment,
    InvestigatorReport,
    ProposedFix,
    RemediationPlan,
    RootCause,
    RootCauseReport,
    SchemaAnalysisReport,
    SchemaChangeAssessment,
)


def _evidence(evidence_id: str, **data) -> Evidence:
    return Evidence(evidence_id=evidence_id, source="tool", tool="t", summary="s", data=data, produced_by="x",
                    created_at="2026-01-01T00:00:00+00:00")


def _finding(fid: str = "F1", ids=("E-Q-001",), rows=None, pct=None, status="confirmed") -> Finding:
    return Finding(finding_id=fid, title="t", issue_type="missing", severity="low", description="d",
                   evidence_ids=list(ids), affected_rows=rows, affected_pct=pct, confidence_estimate=0.5, status=status)


LEDGER = {
    "E-Q-001": _evidence("E-Q-001", affected_rows=35, affected_pct=6.917, details={"null_count": 35}),
    "E-INV-001": _evidence("E-INV-001", segments=[{"issue_rows": 13, "issue_rate_pct": 8.7838}]),
}


class TestFindingGrounding:
    def test_valid_claim_kept(self) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding(rows=35, pct=6.92)])
        assert ground_investigator_report(report, LEDGER) == []
        assert report.findings[0].status == "confirmed" and report.findings[0].affected_rows == 35

    def test_unknown_ids_removed_and_claim_downgraded(self) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding(ids=["E-FAKE-1"])])
        errors = ground_investigator_report(report, LEDGER)
        finding = report.findings[0]
        assert finding.status == "hypothesis" and finding.evidence_ids == []
        assert errors and errors[0].kind == "grounding"

    def test_partial_ids_keep_confirmation(self) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding(ids=["E-Q-001", "E-FAKE-1"])])
        ground_investigator_report(report, LEDGER)
        assert report.findings[0].status == "confirmed" and report.findings[0].evidence_ids == ["E-Q-001"]

    @pytest.mark.parametrize("rows, pct", [(36, None), (None, 50.0), (999, 6.917)])
    def test_unsupported_numbers_removed(self, rows, pct) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding(rows=rows, pct=pct)])
        ground_investigator_report(report, LEDGER)
        finding = report.findings[0]
        assert finding.status == "hypothesis"
        assert any("not found in cited evidence" in n for n in finding.grounding_notes)

    def test_numbers_must_come_from_cited_evidence_only(self) -> None:
        # 13 exists in E-INV-001 but the finding cites only E-Q-001.
        report = InvestigatorReport(summary="s", findings=[_finding(rows=13)])
        ground_investigator_report(report, LEDGER)
        assert report.findings[0].affected_rows is None

    def test_duplicate_finding_ids_are_disambiguated(self) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding("F1"), _finding("F1")])
        ground_investigator_report(report, LEDGER)
        assert len({f.finding_id for f in report.findings}) == 2


class TestSchemaGrounding:
    def test_overwrites_deterministic_facts(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2]})
        contract = SchemaContract.from_dict({"columns": {"a": {"type": "numeric"}, "b": {"type": "text"}}})
        diff = compute_schema_diff(contract, SchemaInferrer().infer(df), df)
        ledger = {"E-S-001": _evidence("E-S-001")}
        report = SchemaAnalysisReport(summary="s", drift_detected=False, confidence_estimate=0.5, assessments=[
            SchemaChangeAssessment(change_type="removed", column="b", compatibility="non_breaking", impact="i",
                                   recommendation="r", evidence_ids=["E-S-001"]),
            SchemaChangeAssessment(change_type="added", column="ghost", compatibility="breaking", impact="i",
                                   recommendation="r", evidence_ids=["E-S-001"]),
        ])
        ground_schema_report(report, ledger, diff)
        assert report.drift_detected is True
        assert report.assessments[0].compatibility == "breaking" and report.assessments[0].status == "confirmed"
        assert report.assessments[1].status == "hypothesis"


class TestRootCauseGrounding:
    def _report(self, **impact) -> RootCauseReport:
        return RootCauseReport(
            summary="s", confidence_estimate=0.5,
            confirmed_facts=[
                Hypothesis(statement="ok", status="confirmed", supporting_evidence_ids=["E-Q-001"], confidence_estimate=0.9),
                Hypothesis(statement="made up", status="confirmed", supporting_evidence_ids=["E-X"], confidence_estimate=0.9),
                Hypothesis(statement="contested", status="confirmed", supporting_evidence_ids=["E-Q-001"],
                           contradicting_evidence_ids=["E-INV-001"], confidence_estimate=0.9),
            ],
            root_causes=[RootCause(cause_id="RC1", category="unknown", explanation="e", related_finding_ids=["F1", "F9"])],
            impact=ImpactAssessment(blast_radius="low", evidence_ids=["E-Q-001"], **impact),
        )

    def test_facts_split_into_confirmed_and_unverified(self) -> None:
        report = self._report(affected_rows_pct=6.917)
        ground_root_cause_report(report, LEDGER, {"F1"})
        assert [f.statement for f in report.confirmed_facts] == ["ok"]
        assert {f.statement for f in report.unverified_claims} == {"made up", "contested"}
        assert report.root_causes[0].related_finding_ids == ["F1"]
        assert report.impact.affected_rows_pct == 6.917

    def test_invented_impact_number_removed(self) -> None:
        report = self._report(affected_rows_pct=42.0)
        ground_root_cause_report(report, LEDGER, {"F1"})
        assert report.impact.affected_rows_pct is None and report.impact.grounding_notes


def _fix(fid: str, findings: list[str]) -> ProposedFix:
    return ProposedFix(fix_id=fid, addresses_finding_ids=findings, title="t", strategy="deduplicate",
                       sql="SELECT DISTINCT * FROM dataset", pyspark="result = df.dropDuplicates()", explanation="e",
                       risk_level="low", reversible=True, expected_effect="e")


def test_remediation_plan_grounding() -> None:
    plan = RemediationPlan(fixes=[_fix("FIX-1", ["F1"]), _fix("FIX-2", ["F9"]), _fix("FIX-1", ["F1"]),
                                  _fix("FIX-3", ["F1"]), _fix("FIX-4", ["F1"])], execution_order=["FIX-3", "FIX-2"])
    errors = ground_remediation_plan(plan, {"F1"}, max_fixes=2)
    assert [f.fix_id for f in plan.fixes] == ["FIX-1", "FIX-3"]
    assert plan.execution_order == ["FIX-3", "FIX-1"]
    assert len(errors) == 3


class TestModels:
    def test_submissions_require_citations(self) -> None:
        payload = {"finding_id": "F1", "title": "t", "issue_type": "missing", "severity": "low", "description": "d",
                   "evidence_ids": [], "confidence_estimate": 0.5}
        with pytest.raises(ValidationError, match="evidence"):
            Finding.model_validate(payload, context=SUBMISSION_CONTEXT)
        assert Finding.model_validate(payload).evidence_ids == []  # stored (grounded) records may be uncited

    def test_confidence_bounds_and_naming(self) -> None:
        assert "confidence_estimate" in Finding.model_fields and "confidence" not in Finding.model_fields
        with pytest.raises(ValidationError):
            _finding().model_copy(update={}).model_validate({**_finding().model_dump(), "confidence_estimate": 1.5})

    def test_checkpoint_round_trip_preserves_nested_types(self) -> None:
        report = InvestigatorReport(summary="s", findings=[_finding(ids=[])])  # uncited after grounding
        serde = make_serializer()
        restored = serde.loads_typed(serde.dumps_typed(report))
        assert isinstance(restored.findings[0], Finding)


def test_evidence_helpers() -> None:
    recorder = EvidenceRecorder("data_investigator", start=next_index([_evidence("E-INV-007")], "INV"))
    assert recorder.record("t", {}, "s", {}).evidence_id == "E-INV-008"
    assert numbers_in({"a": [1, {"b": 2.5}], "c": True, "d": "3"}) == {1.0, 2.5}
    assert number_supported(6.92, [LEDGER["E-Q-001"]]) and not number_supported(7.5, [LEDGER["E-Q-001"]])
