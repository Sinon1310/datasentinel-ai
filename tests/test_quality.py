"""Tests for the data quality rules, engine and scoring."""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
import pytest

from datasentinel.quality import (
    CheckStatus,
    DuplicateRecordsRule,
    InvalidDataTypeRule,
    InvalidDateRule,
    MissingValuesRule,
    OutlierIQRRule,
    QualityConfig,
    QualityDimension,
    QualityEngine,
    QualityRule,
    RuleResult,
    SchemaValidationRule,
    Severity,
    classify_severity,
    compute_scores,
    grade_for,
)
from datasentinel.schema import SchemaContract

CFG = QualityConfig()


def _by_column(results: List[RuleResult]) -> dict:
    return {r.column: r for r in results}


# ---------------------------------------------------------------------- missing values


class TestMissingValues:
    def test_detects_nulls_with_severity(self, make_context) -> None:
        df = pd.DataFrame({
            "complete": range(10),
            "few": [1, None] + list(range(8)),            # 10% -> medium
            "most": [None] * 6 + [1, 2, 3, 4],             # 60% -> critical
        })
        results = _by_column(MissingValuesRule(CFG).evaluate(make_context(df)))
        assert results["complete"].status == CheckStatus.PASSED
        assert results["few"].status == CheckStatus.FAILED
        assert results["few"].affected_rows == 1
        assert results["few"].affected_pct == pytest.approx(10.0)
        assert results["few"].severity == Severity.MEDIUM
        assert results["most"].severity == Severity.CRITICAL

    def test_blank_strings_are_missing(self, make_context) -> None:
        df = pd.DataFrame({"city": ["NY", "  ", "LA", "SF"]})
        result = MissingValuesRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.details["blank_count"] == 1

    def test_tolerance(self, make_context) -> None:
        df = pd.DataFrame({"a": [1, None] + list(range(18))})  # 5%
        result = MissingValuesRule(QualityConfig(max_null_pct=5.0)).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.PASSED


# ---------------------------------------------------------------------- duplicates


class TestDuplicates:
    def test_detects_duplicate_rows(self, make_context) -> None:
        df = pd.DataFrame({"a": [1, 2, 2, 3, 3, 3], "b": ["x", "y", "y", "z", "z", "z"]})
        result = DuplicateRecordsRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.affected_rows == 3
        assert result.details["duplicate_groups"] == 2
        assert result.severity == Severity.CRITICAL  # 50% of rows

    def test_no_duplicates(self, make_context) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "x", "x"]})
        assert DuplicateRecordsRule(CFG).evaluate(make_context(df))[0].status == CheckStatus.PASSED


# ---------------------------------------------------------------------- data types and dates


class TestInvalidTypes:
    def test_flags_unparseable_numeric_strings(self, make_context) -> None:
        df = pd.DataFrame({"qty": [str(i) for i in range(1, 20)] + ["two"]})
        result = InvalidDataTypeRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.affected_rows == 1
        assert result.sample_values == ["two"]
        assert result.severity == Severity.MEDIUM  # 5%

    def test_flags_mixed_python_types(self, make_context) -> None:
        df = pd.DataFrame({"code": ["a", "b", "c", "d", 5]}, dtype=object)
        result = InvalidDataTypeRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.sample_values == [5]

    def test_native_types_pass(self, make_context) -> None:
        df = pd.DataFrame({"n": [1, 2, 3], "f": [True, False, True]})
        results = InvalidDataTypeRule(CFG).evaluate(make_context(df))
        assert all(r.status == CheckStatus.PASSED for r in results)


class TestInvalidDates:
    def test_flags_impossible_unparseable_and_out_of_range(self, make_context) -> None:
        dates = [f"2024-01-{d:02d}" for d in range(1, 21)] + ["2024-02-30", "garbage", "1850-01-01", "2150-01-01"]
        df = pd.DataFrame({"order_date": dates})
        result = InvalidDateRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.details["unparseable"] == 2
        assert result.details["before_min_date"] == 1
        assert result.details["after_max_date"] == 1
        assert result.affected_rows == 4

    def test_skipped_without_date_columns(self, make_context) -> None:
        result = InvalidDateRule(CFG).evaluate(make_context(pd.DataFrame({"a": [1, 2, 3]})))[0]
        assert result.status == CheckStatus.SKIPPED


# ---------------------------------------------------------------------- outliers


class TestOutliers:
    def test_detects_iqr_outliers(self, make_context, normal_values) -> None:
        values = np.append(normal_values, [500.0, -300.0, 900.0])
        df = pd.DataFrame({"amount": values})
        result = OutlierIQRRule(CFG).evaluate(make_context(df))[0]
        assert result.status == CheckStatus.FAILED
        assert result.affected_rows == 3
        assert result.details["extreme_outlier_count"] == 3
        assert result.details["above_upper"] == 2 and result.details["below_lower"] == 1
        assert result.sample_values[0] == 900.0  # most extreme first
        q1, q3 = np.percentile(values, [25, 75])
        assert result.details["upper_fence"] == pytest.approx(q3 + 1.5 * (q3 - q1))

    def test_clean_column_passes(self, make_context, normal_values) -> None:
        result = OutlierIQRRule(CFG).evaluate(make_context(pd.DataFrame({"v": normal_values})))[0]
        assert result.status == CheckStatus.PASSED
        assert result.affected_rows == 0

    def test_known_small_example(self, make_context) -> None:
        df = pd.DataFrame({"v": [10, 12, 11, 13, 12, 11, 10, 12, 100]})
        result = OutlierIQRRule(CFG).evaluate(make_context(df))[0]
        assert result.affected_rows == 1 and result.sample_values == [100.0]

    def test_skips_primary_keys_and_constant_columns(self, make_context) -> None:
        df = pd.DataFrame({"order_id": range(1, 21), "const": [5] * 19 + [6]})
        results = _by_column(OutlierIQRRule(CFG).evaluate(make_context(df)))
        assert results["order_id"].status == CheckStatus.SKIPPED
        assert results["const"].status == CheckStatus.SKIPPED  # IQR == 0

    def test_iqr_multiplier_is_configurable(self, make_context) -> None:
        df = pd.DataFrame({"v": [10, 12, 11, 13, 12, 11, 10, 12, 20]})
        strict = OutlierIQRRule(QualityConfig(iqr_multiplier=1.5)).evaluate(make_context(df))[0]
        # Q1=11, Q3=12, IQR=1 -> a 10x fence puts the upper bound at 22, so 20 is no longer an outlier.
        loose = OutlierIQRRule(QualityConfig(iqr_multiplier=10.0, extreme_iqr_multiplier=12.0)).evaluate(make_context(df))[0]
        assert strict.affected_rows == 1
        assert loose.affected_rows == 0


# ---------------------------------------------------------------------- schema validation


class TestSchemaValidation:
    def test_contract_violations(self, make_context) -> None:
        df = pd.DataFrame({"id": [1, 2, 2], "name": ["a", None, "c"], "extra": [1, 2, 3]})
        contract = SchemaContract.from_dict({
            "allow_extra_columns": False,
            "columns": {
                "id": {"type": "numeric", "unique": True},
                "name": {"type": "categorical", "nullable": False},
                "amount": {"type": "numeric", "required": True},
            },
        })
        results = SchemaValidationRule(CFG).evaluate(make_context(df, contract))
        by_target = {(r.column, r.details.get("check")): r for r in results}
        assert by_target[("amount", "required_column")].severity == Severity.CRITICAL
        assert by_target[("id", "column_contract")].status == CheckStatus.FAILED
        assert by_target[("name", "column_contract")].status == CheckStatus.FAILED
        assert by_target[(None, "unexpected_columns")].status == CheckStatus.FAILED

    def test_contract_type_mismatch(self, make_context) -> None:
        df = pd.DataFrame({"amount": ["a", "b", "c"]})
        contract = SchemaContract.from_dict({"columns": {"amount": {"type": "numeric"}}})
        result = SchemaValidationRule(CFG).evaluate(make_context(df, contract))[0]
        assert result.status == CheckStatus.FAILED and "expected numeric" in result.message

    def test_structural_checks_without_contract(self, make_context) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "const": ["x", "x", "x"], "empty": [None, None, None]})
        results = {r.details["check"]: r for r in SchemaValidationRule(CFG).evaluate(make_context(df))}
        assert results["empty_columns"].status == CheckStatus.FAILED
        assert results["constant_columns"].status == CheckStatus.FAILED
        assert results["primary_key"].status == CheckStatus.PASSED


# ---------------------------------------------------------------------- scoring


def _result(dimension: QualityDimension, status: CheckStatus, severity: Severity = Severity.INFO) -> RuleResult:
    return RuleResult(rule_id="t", rule_name="t", dimension=dimension, status=status, severity=severity, message="")


class TestScoring:
    def test_severity_bands(self) -> None:
        bands = CFG.missing_bands
        assert classify_severity(0.5, bands, Severity.CRITICAL) == Severity.LOW
        assert classify_severity(5.0, bands, Severity.CRITICAL) == Severity.LOW
        assert classify_severity(5.1, bands, Severity.CRITICAL) == Severity.MEDIUM
        assert classify_severity(75, bands, Severity.CRITICAL) == Severity.CRITICAL

    def test_hand_computed_score(self) -> None:
        results = [
            _result(QualityDimension.COMPLETENESS, CheckStatus.PASSED),                 # 100
            _result(QualityDimension.COMPLETENESS, CheckStatus.FAILED, Severity.MEDIUM),  # 50
            _result(QualityDimension.UNIQUENESS, CheckStatus.FAILED, Severity.HIGH),      # 25
            _result(QualityDimension.VALIDITY, CheckStatus.PASSED),                     # 100
            _result(QualityDimension.ACCURACY, CheckStatus.SKIPPED),                    # excluded
        ]
        overall, dims = compute_scores(results, CFG.dimension_weights, CFG.severity_penalties)
        # completeness 75 (w .25), uniqueness 25 (w .20), validity 100 (w .25); active weight .70
        expected = (75 * 0.25 + 25 * 0.20 + 100 * 0.25) / 0.70
        assert overall == pytest.approx(expected, abs=0.01)
        by_dim = {d.dimension: d for d in dims}
        assert by_dim[QualityDimension.COMPLETENESS].score == 75
        assert by_dim[QualityDimension.ACCURACY].score is None
        assert by_dim[QualityDimension.ACCURACY].effective_weight == 0
        assert sum(d.contribution for d in dims) == pytest.approx(overall, abs=0.05)
        assert [r.score for r in results] == [100, 50, 25, 100, None]

    def test_all_passed_scores_100(self) -> None:
        results = [_result(d, CheckStatus.PASSED) for d in QualityDimension]
        overall, _ = compute_scores(results, CFG.dimension_weights, CFG.severity_penalties)
        assert overall == 100

    def test_nothing_evaluated(self) -> None:
        overall, _ = compute_scores([_result(QualityDimension.VALIDITY, CheckStatus.SKIPPED)],
                                    CFG.dimension_weights, CFG.severity_penalties)
        assert overall is None and grade_for(overall) == "N/A"

    @pytest.mark.parametrize("score, grade", [(100, "A"), (90, "A"), (89.99, "B"), (72, "C"), (60, "D"), (10, "F")])
    def test_grades(self, score: float, grade: str) -> None:
        assert grade_for(score) == grade


# ---------------------------------------------------------------------- engine and pipeline


class TestEngine:
    def test_clean_dataset_scores_high(self, analyze, normal_values) -> None:
        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(200)],
            "amount": normal_values,
            "region": np.resize(["N", "S", "E", "W"], 200),
            "signup_date": pd.date_range("2024-01-01", periods=200, freq="D").strftime("%Y-%m-%d"),
        })
        report = analyze(df).quality
        assert report.overall_score == 100
        assert report.grade == "A"
        assert report.failed_results == []

    def test_dirty_dataset_scores_lower_and_ranks_issues(self, analyze, normal_values) -> None:
        df = pd.DataFrame({"id": range(200), "amount": normal_values})
        df.loc[0:59, "amount"] = np.nan  # 30% missing -> high
        df = pd.concat([df, df.iloc[100:110]], ignore_index=True)  # duplicates
        report = analyze(df).quality
        assert report.overall_score < 80
        assert report.failed_results[0].severity.rank >= report.failed_results[-1].severity.rank
        assert {r.rule_id for r in report.failed_results} >= {
            "completeness.missing_values", "uniqueness.duplicate_records"}

    def test_faulty_rule_is_isolated(self, make_context) -> None:
        class BrokenRule(QualityRule):
            rule_id, name, dimension, description = "x.broken", "Broken", QualityDimension.VALIDITY, ""

            def evaluate(self, ctx):
                raise RuntimeError("boom")

        engine = QualityEngine(CFG, rules=[BrokenRule(CFG), MissingValuesRule(CFG)])
        report = engine.run(make_context(pd.DataFrame({"a": [1, 2]})))
        broken = next(r for r in report.results if r.rule_id == "x.broken")
        assert broken.status == CheckStatus.SKIPPED and "boom" in broken.message
        assert report.overall_score == 100

    def test_report_serializes(self, analyze) -> None:
        report = analyze(pd.DataFrame({"a": [1, 2, None], "d": ["2024-01-01", "bad", "2024-01-03"]})).quality
        data = json.loads(report.to_json())
        assert data["summary"]["total_checks"] == len(report.results)
        assert "methodology" in data
        assert len(report.results_dataframe()) == len(report.results)


def test_sample_dataset_end_to_end() -> None:
    """The bundled sample has known injected issues; each must be detected."""
    from pathlib import Path

    from datasentinel.pipeline import analyze_file

    path = Path(__file__).resolve().parents[1] / "sample_data" / "customer_orders.csv"
    if not path.exists():
        pytest.skip("Sample data not generated")
    report = analyze_file(path).quality
    failed = {(r.rule_id, r.column) for r in report.failed_results}
    assert ("uniqueness.duplicate_records", None) in failed
    assert ("validity.dates", "order_date") in failed
    assert ("validity.data_types", "quantity") in failed
    assert ("accuracy.outliers_iqr", "quantity") in failed
    assert ("completeness.missing_values", "customer_email") in failed
