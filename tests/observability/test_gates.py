"""Quality gates: the approved formula, its boundaries, configuration and the real quality engine."""

from __future__ import annotations

import pytest

from datasentinel.config import Settings
from datasentinel.exceptions import QualityGateError
from datasentinel.ingestion import DatasetLoader
from datasentinel.observability import (
    FailureCategory, FailureClassifier, GateConfig, enforce_quality_gate, evaluate_quality_gate,
)
from datasentinel.pipeline import analyze_dataset
from datasentinel.quality.models import CheckStatus, QualityDimension, QualityReport, RuleResult, Severity


def _result(severity: Severity, status: CheckStatus = CheckStatus.FAILED, rule: str = "r", column: str = "c",
            pct: float = 55.0) -> RuleResult:
    return RuleResult(rule, rule, QualityDimension.COMPLETENESS, status, severity, "m", column=column,
                      affected_pct=pct)


def _report(score, results=()) -> QualityReport:
    return QualityReport("d.csv", "2026-11-23T00:00:00+00:00", score, "N/A" if score is None else "C", [],
                         list(results))


@pytest.mark.parametrize("score,passed", [(59.9, False), (59.99, False), (60.0, True), (60.1, True), (100.0, True),
                                          (0.0, False)])
def test_score_boundary(score: float, passed: bool) -> None:
    result = evaluate_quality_gate(_report(score))
    assert result.passed is passed
    assert bool(result.reasons) is not passed


def test_any_failed_critical_check_fails_the_gate() -> None:
    result = evaluate_quality_gate(_report(95.0, [_result(Severity.CRITICAL, rule="completeness.missing_values")]))
    assert not result.passed and result.reasons == ("1 critical quality check(s) failed",)
    assert result.critical_failures[0].rule_id == "completeness.missing_values"


@pytest.mark.parametrize("severity,status", [(Severity.HIGH, CheckStatus.FAILED), (Severity.CRITICAL, CheckStatus.PASSED),
                                             (Severity.CRITICAL, CheckStatus.SKIPPED)])
def test_only_failed_critical_checks_count(severity, status) -> None:
    assert evaluate_quality_gate(_report(95.0, [_result(severity, status)])).passed


def test_unscored_reports_fail_by_default_and_can_be_allowed() -> None:
    assert evaluate_quality_gate(_report(None)).reasons == ("no quality checks could be evaluated (unscored)",)
    assert evaluate_quality_gate(_report(None), GateConfig(fail_when_unscored=False)).passed


def test_all_reasons_are_reported_together() -> None:
    result = evaluate_quality_gate(_report(40.0, [_result(Severity.CRITICAL)]))
    assert len(result.reasons) == 2


def test_configuration() -> None:
    assert evaluate_quality_gate(_report(70.0), GateConfig(min_score=75)).passed is False
    lenient = GateConfig(fail_on_critical=False)
    result = evaluate_quality_gate(_report(95.0, [_result(Severity.CRITICAL)]), lenient)
    assert result.passed and result.critical_failures == ()
    for bad in (-1, 100.5, True):
        with pytest.raises(ValueError):
            GateConfig(min_score=bad)
    settings = Settings(_env_file=None, quality_gate_min_score=80, quality_gate_fail_on_critical=False,
                        quality_gate_fail_when_unscored=False)
    assert GateConfig.from_settings(settings) == GateConfig(80.0, False, False)


def test_enforce_raises_a_classified_error_with_the_result() -> None:
    report = _report(58.7)
    with pytest.raises(QualityGateError, match="58.70 is below the minimum 60.00") as info:
        enforce_quality_gate(report)
    assert info.value.gate_result.overall_score == 58.7
    assert FailureClassifier().classify(info.value).category is FailureCategory.DATA_QUALITY
    assert enforce_quality_gate(_report(90.0)).passed


def test_context_is_json_metadata_only() -> None:
    context = evaluate_quality_gate(_report(40.0, [_result(Severity.CRITICAL, column="email")])).to_context()
    assert context["critical_failures"] == [{"rule_id": "r", "target": "email", "affected_pct": 55.0}]
    assert set(context) == {"passed", "reasons", "overall_score", "grade", "critical_failures", "min_score",
                            "fail_on_critical", "fail_when_unscored"}


def test_gate_on_real_engine_results() -> None:
    clean = b"id,amount\n1,10\n2,20\n3,30\n4,40\n5,50\n"
    assert evaluate_quality_gate(analyze_dataset(DatasetLoader().load_bytes(clean, "clean.csv")).quality).passed
    mostly_missing = b"id,amount,note\n" + b"".join(f"{i},,\n".encode() for i in range(1, 21))
    result = evaluate_quality_gate(analyze_dataset(DatasetLoader().load_bytes(mostly_missing, "bad.csv")).quality)
    assert not result.passed
    assert any(c.rule_id.startswith("completeness") for c in result.critical_failures)
