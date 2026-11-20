"""Thresholds and weights for the data quality engine.

Every number that influences a pass/fail decision, a severity or the overall score
lives here, so the scoring is fully transparent and auditable — and can be tuned per
dataset without code changes.

Severity bands are tuples of ``(upper_bound_pct, severity)`` evaluated in order: the
first band whose upper bound is >= the affected percentage wins; anything above the
last band gets the rule's ``*_above`` severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from datasentinel.quality.models import QualityDimension, Severity

SeverityBands = Tuple[Tuple[float, Severity], ...]


def classify_severity(pct: float, bands: SeverityBands, above: Severity) -> Severity:
    """Map an affected percentage to a severity using ordered bands."""
    for upper, severity in bands:
        if pct <= upper:
            return severity
    return above


@dataclass
class QualityConfig:
    """Configuration for all built-in quality rules and the scoring model."""

    # Completeness — missing values per column.
    max_null_pct: float = 0.0
    missing_bands: SeverityBands = ((5.0, Severity.LOW), (20.0, Severity.MEDIUM), (50.0, Severity.HIGH))
    missing_above: Severity = Severity.CRITICAL

    # Uniqueness — fully duplicated rows.
    max_duplicate_pct: float = 0.0
    duplicate_bands: SeverityBands = ((1.0, Severity.LOW), (5.0, Severity.MEDIUM), (20.0, Severity.HIGH))
    duplicate_above: Severity = Severity.CRITICAL

    # Validity — values that do not match the inferred column type.
    invalid_type_bands: SeverityBands = ((1.0, Severity.LOW), (5.0, Severity.MEDIUM), (20.0, Severity.HIGH))
    invalid_type_above: Severity = Severity.CRITICAL

    # Validity — unparseable or out-of-range dates.
    min_valid_date: str = "1900-01-01"
    max_future_days: int = 365 * 5
    invalid_date_bands: SeverityBands = ((1.0, Severity.LOW), (5.0, Severity.MEDIUM), (20.0, Severity.HIGH))
    invalid_date_above: Severity = Severity.CRITICAL

    # Accuracy — IQR (Tukey fence) outliers.
    iqr_multiplier: float = 1.5
    extreme_iqr_multiplier: float = 3.0
    outlier_tolerance_pct: float = 1.0
    min_values_for_outliers: int = 8
    outlier_bands: SeverityBands = ((5.0, Severity.LOW), (15.0, Severity.MEDIUM))
    outlier_above: Severity = Severity.HIGH

    # Reporting.
    sample_size: int = 5

    # Scoring model.
    dimension_weights: Dict[QualityDimension, float] = field(
        default_factory=lambda: {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.UNIQUENESS: 0.20,
            QualityDimension.VALIDITY: 0.25,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.ACCURACY: 0.15,
        }
    )
    severity_penalties: Dict[Severity, float] = field(
        default_factory=lambda: {
            Severity.INFO: 0.0,
            Severity.LOW: 0.25,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.75,
            Severity.CRITICAL: 1.0,
        }
    )
