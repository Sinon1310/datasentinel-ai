"""Data models for quality rule results and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from datasentinel.utils.serialization import to_builtin


class Severity(str, Enum):
    """Impact of a failed check. Passed checks carry ``INFO``."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Ordinal rank, higher is more severe."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}


class CheckStatus(str, Enum):
    """Outcome of a single check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class QualityDimension(str, Enum):
    """Data quality dimensions used to group checks and weight the score."""

    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"


@dataclass
class RuleResult:
    """Result of evaluating one rule against one target (a column or the whole dataset).

    Attributes:
        rule_id: Stable identifier, e.g. ``completeness.missing_values``.
        rule_name: Human-readable rule name.
        dimension: Quality dimension the rule contributes to.
        status: passed / failed / skipped.
        severity: Impact if failed (``INFO`` for passed / skipped checks).
        message: One-line explanation of the outcome.
        column: Target column, or ``None`` for dataset-level checks.
        affected_rows: Number of rows / values violating the rule.
        affected_pct: ``affected_rows`` as a percentage of the evaluated population.
        threshold: The tolerance the metric was compared against, if any.
        details: Rule-specific evidence (bounds, counts, sub-check names, ...).
        sample_values: Example offending values.
        score: Score contribution in [0, 100], filled in by the scoring stage.
    """

    rule_id: str
    rule_name: str
    dimension: QualityDimension
    status: CheckStatus
    severity: Severity
    message: str
    column: Optional[str] = None
    affected_rows: int = 0
    affected_pct: float = 0.0
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)
    score: Optional[float] = None

    @property
    def passed(self) -> bool:
        """True when the check passed."""
        return self.status == CheckStatus.PASSED

    @property
    def target(self) -> str:
        """Column name, or ``(dataset)`` for dataset-level checks."""
        return self.column if self.column is not None else "(dataset)"


@dataclass
class DimensionScore:
    """Score for one quality dimension and its contribution to the overall score."""

    dimension: QualityDimension
    weight: float
    effective_weight: float
    score: Optional[float]
    checks_evaluated: int
    checks_failed: int
    contribution: float


@dataclass
class QualityReport:
    """Complete quality assessment of a dataset."""

    dataset_name: str
    generated_at: str
    overall_score: Optional[float]
    grade: str
    dimension_scores: List[DimensionScore]
    results: List[RuleResult]
    methodology: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed_results(self) -> List[RuleResult]:
        """Failed checks ordered from most to least severe."""
        failed = [r for r in self.results if r.status == CheckStatus.FAILED]
        return sorted(failed, key=lambda r: (-r.severity.rank, -r.affected_pct, r.rule_id))

    def summary(self) -> Dict[str, Any]:
        """Counts of checks by status and failed checks by severity."""
        by_status = {s.value: 0 for s in CheckStatus}
        by_severity = {s.value: 0 for s in Severity if s != Severity.INFO}
        for result in self.results:
            by_status[result.status.value] += 1
            if result.status == CheckStatus.FAILED:
                by_severity[result.severity.value] += 1
        return {"total_checks": len(self.results), "by_status": by_status, "failed_by_severity": by_severity}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary."""
        data = to_builtin(self)
        data["summary"] = self.summary()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def results_dataframe(self) -> pd.DataFrame:
        """Tabular view of all rule results."""
        rows = [
            {
                "rule_id": r.rule_id,
                "rule": r.rule_name,
                "dimension": r.dimension.value,
                "target": r.target,
                "status": r.status.value,
                "severity": r.severity.value,
                "affected_rows": r.affected_rows,
                "affected_pct": round(r.affected_pct, 2),
                "score": None if r.score is None else round(r.score, 1),
                "message": r.message,
            }
            for r in self.results
        ]
        return pd.DataFrame(rows)
