"""Quality engine: runs rules, isolates rule failures and builds the report."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from datasentinel.quality.config import QualityConfig
from datasentinel.quality.models import CheckStatus, QualityReport, RuleResult, Severity
from datasentinel.quality.rules import QualityContext, QualityRule, default_rules
from datasentinel.quality.scoring import compute_scores, grade_for

logger = logging.getLogger(__name__)


class QualityEngine:
    """Run a set of :class:`QualityRule` objects and produce a :class:`QualityReport`.

    A rule that raises an unexpected exception does not abort the run; it is recorded
    as a *skipped* result carrying the error message, and the exception is logged.

    Args:
        config: Thresholds and scoring weights. Defaults to :class:`QualityConfig`.
        rules: Custom rule set. Defaults to :func:`default_rules`.
    """

    def __init__(self, config: Optional[QualityConfig] = None, rules: Optional[Sequence[QualityRule]] = None) -> None:
        self.config = config or QualityConfig()
        self.rules: List[QualityRule] = list(rules) if rules is not None else default_rules(self.config)

    def run(self, ctx: QualityContext, dataset_name: str = "dataset") -> QualityReport:
        """Evaluate all rules against ``ctx``."""
        results: List[RuleResult] = []
        for rule in self.rules:
            try:
                rule_results = rule.evaluate(ctx)
            except Exception as exc:  # noqa: BLE001 — isolate faulty rules by design
                logger.exception("Quality rule '%s' failed", rule.rule_id)
                rule_results = [
                    RuleResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        dimension=rule.dimension,
                        status=CheckStatus.SKIPPED,
                        severity=Severity.INFO,
                        message=f"Rule could not be evaluated: {exc}",
                        details={"error": type(exc).__name__},
                    )
                ]
            results.extend(rule_results)

        overall, dimension_scores = compute_scores(
            results, self.config.dimension_weights, self.config.severity_penalties
        )
        report = QualityReport(
            dataset_name=dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            overall_score=overall,
            grade=grade_for(overall),
            dimension_scores=dimension_scores,
            results=results,
            methodology={
                "check_score": "100 x (1 - severity_penalty); passed = 100; skipped checks excluded",
                "dimension_score": "mean of check scores in the dimension",
                "overall_score": "weighted mean of dimension scores; weights re-normalised over evaluated dimensions",
                "severity_penalties": {k.value: v for k, v in self.config.severity_penalties.items()},
                "dimension_weights": {k.value: v for k, v in self.config.dimension_weights.items()},
                "grades": {"A": ">= 90", "B": ">= 80", "C": ">= 70", "D": ">= 60", "F": "< 60"},
            },
        )
        summary = report.summary()
        logger.info(
            "Quality assessment for '%s': score=%s grade=%s checks=%d failed=%d",
            dataset_name, overall, report.grade, summary["total_checks"], summary["by_status"]["failed"],
        )
        return report
