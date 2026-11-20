"""Deterministic data quality rules.

Each rule is a small class with a single responsibility. Rules receive an immutable
:class:`QualityContext` and return a list of :class:`RuleResult` objects — one per
evaluated column, or one for dataset-level checks. Adding a rule means subclassing
:class:`QualityRule` and registering it in :func:`default_rules`; the engine and scoring
pick it up automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from datasentinel.profiling.models import DatasetProfile
from datasentinel.quality.config import QualityConfig, classify_severity
from datasentinel.quality.models import CheckStatus, QualityDimension, RuleResult, Severity
from datasentinel.schema.contract import SchemaContract
from datasentinel.schema.models import SchemaReport, SemanticType
from datasentinel.utils.parsing import looks_like_date_column, missing_mask, parse_datetime

logger = logging.getLogger(__name__)

_STRING_FAMILY = {SemanticType.CATEGORICAL, SemanticType.TEXT}


@dataclass(frozen=True)
class QualityContext:
    """Everything a rule may inspect."""

    raw: pd.DataFrame
    typed: pd.DataFrame
    schema: SchemaReport
    profile: DatasetProfile
    contract: Optional[SchemaContract] = None


def _pct(part: int, whole: int) -> float:
    return (part / whole * 100.0) if whole else 0.0


def _samples(values: pd.Series, limit: int) -> List[Any]:
    return values.drop_duplicates().head(limit).tolist()


class QualityRule(ABC):
    """Base class for quality rules."""

    rule_id: ClassVar[str]
    name: ClassVar[str]
    dimension: ClassVar[QualityDimension]
    description: ClassVar[str]

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    @abstractmethod
    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        """Evaluate the rule and return its results."""

    def result(
        self,
        status: CheckStatus,
        message: str,
        severity: Severity = Severity.INFO,
        column: Optional[str] = None,
        **kwargs: Any,
    ) -> RuleResult:
        """Construct a :class:`RuleResult` pre-filled with this rule's identity."""
        if status != CheckStatus.FAILED:
            severity = Severity.INFO
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.name,
            dimension=self.dimension,
            status=status,
            severity=severity,
            message=message,
            column=column,
            **kwargs,
        )


# ---------------------------------------------------------------------- completeness


class MissingValuesRule(QualityRule):
    """Flags columns whose share of missing values (null or blank) exceeds the tolerance."""

    rule_id = "completeness.missing_values"
    name = "Missing values"
    dimension = QualityDimension.COMPLETENESS
    description = "Share of null or whitespace-only values per column."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        cfg = self.config
        n_rows = ctx.schema.n_rows
        results = []
        for column in ctx.schema.columns:
            mask = missing_mask(ctx.raw[column.name])
            missing = int(mask.sum())
            pct = _pct(missing, n_rows)
            details = {
                "null_count": int(ctx.raw[column.name].isna().sum()),
                "blank_count": int(missing - ctx.raw[column.name].isna().sum()),
                "sample_row_indices": mask[mask].index[: cfg.sample_size].tolist(),
            }
            common = dict(column=column.name, affected_rows=missing, affected_pct=pct,
                          threshold=cfg.max_null_pct, details=details)
            if missing == 0:
                results.append(self.result(CheckStatus.PASSED, "No missing values", **common))
            elif pct <= cfg.max_null_pct:
                results.append(self.result(
                    CheckStatus.PASSED, f"{missing} missing value(s) ({pct:.2f}%) within tolerance", **common))
            else:
                severity = classify_severity(pct, cfg.missing_bands, cfg.missing_above)
                results.append(self.result(
                    CheckStatus.FAILED, f"{missing} missing value(s) ({pct:.2f}% of rows)", severity, **common))
        return results


# ---------------------------------------------------------------------- uniqueness


class DuplicateRecordsRule(QualityRule):
    """Flags fully duplicated rows."""

    rule_id = "uniqueness.duplicate_records"
    name = "Duplicate records"
    dimension = QualityDimension.UNIQUENESS
    description = "Rows that are exact copies of an earlier row."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        cfg = self.config
        duplicated = ctx.raw.duplicated(keep="first")
        count = int(duplicated.sum())
        pct = _pct(count, len(ctx.raw))
        groups = int(ctx.raw[ctx.raw.duplicated(keep=False)].drop_duplicates().shape[0]) if count else 0
        details = {"duplicate_groups": groups, "sample_row_indices": duplicated[duplicated].index[: cfg.sample_size].tolist()}
        common = dict(affected_rows=count, affected_pct=pct, threshold=cfg.max_duplicate_pct, details=details)
        if count == 0:
            return [self.result(CheckStatus.PASSED, "No duplicate rows", **common)]
        if pct <= cfg.max_duplicate_pct:
            return [self.result(CheckStatus.PASSED, f"{count} duplicate row(s) within tolerance", **common)]
        severity = classify_severity(pct, cfg.duplicate_bands, cfg.duplicate_above)
        return [self.result(
            CheckStatus.FAILED,
            f"{count} duplicate row(s) ({pct:.2f}%) across {groups} group(s)",
            severity,
            **common,
        )]


# ---------------------------------------------------------------------- validity


class InvalidDataTypeRule(QualityRule):
    """Flags values that do not conform to the column's inferred type.

    * Numeric / boolean columns parsed from text: values that fail to parse.
    * Text-like columns: values whose Python type differs from the dominant type
      (e.g. a JSON field mixing numbers and strings).

    Datetime columns are handled by :class:`InvalidDateRule`.
    """

    rule_id = "validity.data_types"
    name = "Invalid data types"
    dimension = QualityDimension.VALIDITY
    description = "Values that cannot be interpreted as the column's inferred type."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        cfg = self.config
        results = []
        for column in ctx.schema.columns:
            stype = column.semantic_type
            if stype in (SemanticType.EMPTY, SemanticType.DATETIME):
                continue
            raw = ctx.raw[column.name]
            present = ~missing_mask(raw)
            n_present = int(present.sum())

            if stype in (SemanticType.NUMERIC, SemanticType.BOOLEAN):
                invalid = present & ctx.typed[column.name].isna()
                expected = f"{stype.value} ({column.storage_type})" if not column.inferred_from_string else stype.value
            else:
                type_names = raw[present].map(lambda v: type(v).__name__)
                dominant = type_names.mode().iloc[0] if not type_names.empty else "str"
                invalid = pd.Series(False, index=raw.index)
                invalid.loc[type_names.index] = type_names != dominant
                expected = f"{stype.value} ({dominant} values)"

            count = int(invalid.sum())
            pct = _pct(count, n_present)
            common = dict(
                column=column.name, affected_rows=count, affected_pct=pct, threshold=0.0,
                details={"expected_type": expected, "values_checked": n_present},
                sample_values=_samples(raw[invalid], cfg.sample_size),
            )
            if count == 0:
                results.append(self.result(CheckStatus.PASSED, f"All values are valid {expected}", **common))
            else:
                severity = classify_severity(pct, cfg.invalid_type_bands, cfg.invalid_type_above)
                results.append(self.result(
                    CheckStatus.FAILED, f"{count} value(s) ({pct:.2f}%) are not valid {expected}", severity, **common))
        return results


class InvalidDateRule(QualityRule):
    """Flags unparseable, impossible (e.g. 2023-02-30) or out-of-range dates.

    Evaluates datetime columns plus text columns whose *name* suggests dates
    (e.g. ``created_at``) but which failed datetime inference.
    """

    rule_id = "validity.dates"
    name = "Invalid dates"
    dimension = QualityDimension.VALIDITY
    description = "Date values that cannot be parsed or fall outside the plausible range."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        cfg = self.config
        min_date = pd.Timestamp(cfg.min_valid_date)
        max_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=cfg.max_future_days)
        candidates = [
            c for c in ctx.schema.columns
            if c.semantic_type == SemanticType.DATETIME
            or (c.semantic_type in _STRING_FAMILY and looks_like_date_column(c.name))
        ]
        if not candidates:
            return [self.result(CheckStatus.SKIPPED, "No datetime columns detected")]

        results = []
        for column in candidates:
            raw = ctx.raw[column.name]
            typed = ctx.typed[column.name]
            parsed = typed if is_datetime64_any_dtype(typed.dtype) else parse_datetime(raw)
            present = ~missing_mask(raw)
            n_present = int(present.sum())
            unparseable = present & parsed.isna()
            too_early = parsed < min_date
            too_late = parsed > max_date
            invalid = unparseable | too_early.fillna(False) | too_late.fillna(False)
            count = int(invalid.sum())
            pct = _pct(count, n_present)
            details = {
                "unparseable": int(unparseable.sum()),
                "before_min_date": int(too_early.sum()),
                "after_max_date": int(too_late.sum()),
                "valid_range": [min_date.date().isoformat(), max_date.date().isoformat()],
                "inferred_as_datetime": column.semantic_type == SemanticType.DATETIME,
            }
            common = dict(column=column.name, affected_rows=count, affected_pct=pct, threshold=0.0,
                          details=details, sample_values=[str(v) for v in _samples(raw[invalid], cfg.sample_size)])
            if count == 0:
                results.append(self.result(CheckStatus.PASSED, "All dates are valid and within range", **common))
            else:
                parts = [f"{v} {k.replace('_', ' ')}" for k, v in
                         (("unparseable", details["unparseable"]), ("before_min_date", details["before_min_date"]),
                          ("after_max_date", details["after_max_date"])) if v]
                severity = classify_severity(pct, cfg.invalid_date_bands, cfg.invalid_date_above)
                results.append(self.result(
                    CheckStatus.FAILED, f"{count} invalid date(s) ({pct:.2f}%): {', '.join(parts)}", severity, **common))
        return results


# ---------------------------------------------------------------------- accuracy


class OutlierIQRRule(QualityRule):
    """Detects outliers in numeric columns using Tukey's IQR fences.

    * Mild outliers: outside ``[Q1 - k*IQR, Q3 + k*IQR]`` (k = ``iqr_multiplier``, 1.5).
    * Extreme outliers: outside the fences with ``extreme_iqr_multiplier`` (3.0).

    A column passes if it has no extreme outliers and its mild-outlier share is within
    ``outlier_tolerance_pct`` — a small tail is expected in most real distributions.
    Primary-key candidates and binary 0/1 columns are excluded.
    """

    rule_id = "accuracy.outliers_iqr"
    name = "Outliers (IQR)"
    dimension = QualityDimension.ACCURACY
    description = "Numeric values far outside the interquartile range."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        cfg = self.config
        key_columns = {c for pk in ctx.schema.primary_key_candidates for c in pk.columns}
        numeric = [c for c in ctx.schema.columns if c.semantic_type == SemanticType.NUMERIC]
        if not numeric:
            return [self.result(CheckStatus.SKIPPED, "No numeric columns detected")]

        results = []
        for column in numeric:
            if column.name in key_columns:
                results.append(self.result(CheckStatus.SKIPPED, "Skipped: primary-key candidate", column=column.name))
                continue
            values = pd.to_numeric(ctx.typed[column.name], errors="coerce").dropna().astype(float)
            values = values[np.isfinite(values.to_numpy())]
            if set(values.unique().tolist()) <= {0.0, 1.0}:
                results.append(self.result(CheckStatus.SKIPPED, "Skipped: binary 0/1 column", column=column.name))
                continue
            if len(values) < cfg.min_values_for_outliers:
                results.append(self.result(
                    CheckStatus.SKIPPED, f"Skipped: fewer than {cfg.min_values_for_outliers} values", column=column.name))
                continue
            results.append(self._evaluate_column(column.name, values))
        return results

    def _evaluate_column(self, name: str, values: pd.Series) -> RuleResult:
        cfg = self.config
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return self.result(
                CheckStatus.SKIPPED, "Skipped: IQR is zero (at least half the values are identical)",
                column=name, details={"q1": q1, "q3": q3, "iqr": iqr},
            )
        lower, upper = q1 - cfg.iqr_multiplier * iqr, q3 + cfg.iqr_multiplier * iqr
        ext_lower, ext_upper = q1 - cfg.extreme_iqr_multiplier * iqr, q3 + cfg.extreme_iqr_multiplier * iqr
        outliers = values[(values < lower) | (values > upper)]
        extreme = values[(values < ext_lower) | (values > ext_upper)]
        count, pct = int(outliers.size), _pct(int(outliers.size), int(values.size))
        median = float(values.median())
        most_extreme = outliers.reindex((outliers - median).abs().sort_values(ascending=False).index)
        details = {
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_fence": lower, "upper_fence": upper,
            "extreme_lower_fence": ext_lower, "extreme_upper_fence": ext_upper,
            "outlier_count": count, "extreme_outlier_count": int(extreme.size),
            "below_lower": int((values < lower).sum()), "above_upper": int((values > upper).sum()),
            "values_checked": int(values.size),
        }
        common = dict(column=name, affected_rows=count, affected_pct=pct, threshold=cfg.outlier_tolerance_pct,
                      details=details, sample_values=most_extreme.head(cfg.sample_size).tolist())
        if count == 0:
            return self.result(CheckStatus.PASSED, f"No outliers outside [{lower:.4g}, {upper:.4g}]", **common)
        if extreme.empty and pct <= cfg.outlier_tolerance_pct:
            return self.result(
                CheckStatus.PASSED, f"{count} mild outlier(s) ({pct:.2f}%) within tolerance, none extreme", **common)
        severity = classify_severity(pct, cfg.outlier_bands, cfg.outlier_above)
        return self.result(
            CheckStatus.FAILED,
            f"{count} outlier(s) ({pct:.2f}%) outside [{lower:.4g}, {upper:.4g}], {extreme.size} extreme",
            severity,
            **common,
        )


# ---------------------------------------------------------------------- consistency


class SchemaValidationRule(QualityRule):
    """Validates the dataset structure.

    With a :class:`SchemaContract`: checks required columns, types, nullability,
    uniqueness and unexpected columns. Without one: runs structural checks (empty
    columns, constant columns, malformed headers, presence of a primary key).
    """

    rule_id = "consistency.schema"
    name = "Schema validation"
    dimension = QualityDimension.CONSISTENCY
    description = "Dataset structure matches the expected schema (or structural best practices)."

    def evaluate(self, ctx: QualityContext) -> List[RuleResult]:
        if ctx.contract is not None:
            return self._validate_contract(ctx, ctx.contract)
        return self._structural_checks(ctx)

    # -- contract mode -------------------------------------------------------

    def _validate_contract(self, ctx: QualityContext, contract: SchemaContract) -> List[RuleResult]:
        results: List[RuleResult] = []
        actual = {c.name: c for c in ctx.schema.columns}
        n_rows = ctx.schema.n_rows

        for name, expected in contract.columns.items():
            if name not in actual:
                if expected.required:
                    results.append(self.result(
                        CheckStatus.FAILED, "Required column is missing", Severity.CRITICAL, column=name,
                        details={"check": "required_column"}))
                else:
                    results.append(self.result(
                        CheckStatus.SKIPPED, "Optional column not present", column=name,
                        details={"check": "required_column"}))
                continue

            column = actual[name]
            violations: List[str] = []
            severity = Severity.INFO
            affected = 0
            if expected.type is not None and column.semantic_type != SemanticType.EMPTY:
                compatible = column.semantic_type == expected.type or (
                    column.semantic_type in _STRING_FAMILY and expected.type in _STRING_FAMILY)
                if not compatible:
                    violations.append(f"type is {column.semantic_type.value}, expected {expected.type.value}")
                    severity = Severity.HIGH
            if not expected.nullable and column.null_count > 0:
                violations.append(f"{column.null_count} null value(s) in non-nullable column")
                severity = Severity.HIGH
                affected = max(affected, column.null_count)
            if expected.unique:
                dupes = int(ctx.raw[name].dropna().duplicated().sum())
                if dupes:
                    violations.append(f"{dupes} duplicate value(s) in unique column")
                    severity = Severity.HIGH
                    affected = max(affected, dupes)

            details = {
                "check": "column_contract",
                "expected": {"type": expected.type.value if expected.type else None, "nullable": expected.nullable,
                             "unique": expected.unique},
                "actual": {"type": column.semantic_type.value, "null_count": column.null_count},
            }
            if violations:
                results.append(self.result(
                    CheckStatus.FAILED, "; ".join(violations).capitalize(), severity, column=name,
                    affected_rows=affected, affected_pct=_pct(affected, n_rows), details=details))
            else:
                results.append(self.result(CheckStatus.PASSED, "Matches contract", column=name, details=details))

        extra = [c for c in actual if c not in contract.columns]
        if extra and not contract.allow_extra_columns:
            results.append(self.result(
                CheckStatus.FAILED, f"{len(extra)} unexpected column(s) not in contract", Severity.MEDIUM,
                details={"check": "unexpected_columns", "columns": extra}, sample_values=extra[: self.config.sample_size]))
        else:
            results.append(self.result(
                CheckStatus.PASSED,
                "No unexpected columns" if not extra else f"{len(extra)} extra column(s) allowed by contract",
                details={"check": "unexpected_columns", "columns": extra}))
        return results

    # -- structural mode -----------------------------------------------------

    def _structural_checks(self, ctx: QualityContext) -> List[RuleResult]:
        schema = ctx.schema
        results: List[RuleResult] = []

        empty = schema.columns_of_type(SemanticType.EMPTY)
        results.append(self.result(
            CheckStatus.FAILED if empty else CheckStatus.PASSED,
            f"{len(empty)} column(s) contain no values" if empty else "No fully empty columns",
            Severity.MEDIUM, details={"check": "empty_columns", "columns": empty}, sample_values=empty[:5]))

        constant = [c.name for c in schema.columns if c.unique_count == 1 and schema.n_rows > 1]
        results.append(self.result(
            CheckStatus.FAILED if constant else CheckStatus.PASSED,
            f"{len(constant)} constant column(s) carry no information" if constant else "No constant columns",
            Severity.LOW, details={"check": "constant_columns", "columns": constant}, sample_values=constant[:5]))

        malformed = [c.name for c in schema.columns
                     if c.name.startswith("Unnamed:") or c.name.startswith("column_") or c.name != c.name.strip()]
        results.append(self.result(
            CheckStatus.FAILED if malformed else CheckStatus.PASSED,
            f"{len(malformed)} column(s) have missing or generated names" if malformed else "All column names are well-formed",
            Severity.LOW, details={"check": "column_names", "columns": malformed}, sample_values=malformed[:5]))

        pk = schema.primary_key
        results.append(self.result(
            CheckStatus.PASSED if pk else CheckStatus.FAILED,
            f"Primary key candidate identified: {pk.label}" if pk else "No column or column pair uniquely identifies rows",
            Severity.LOW, details={"check": "primary_key", "candidate": pk.label if pk else None}))
        return results


def default_rules(config: QualityConfig) -> List[QualityRule]:
    """The built-in rule set, in evaluation order."""
    return [
        MissingValuesRule(config),
        DuplicateRecordsRule(config),
        InvalidDataTypeRule(config),
        InvalidDateRule(config),
        OutlierIQRRule(config),
        SchemaValidationRule(config),
    ]
