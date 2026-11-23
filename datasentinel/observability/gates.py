"""Deterministic quality gates for pipelines, built on the engine's real quality models.

A gate fails when ANY of these holds (approved decision 6, decision I1)::

    fail_on_critical  and  some check has status FAILED and severity CRITICAL
    overall_score is not None  and  overall_score < min_score           (default 60.0; 60.0 itself passes)
    overall_score is None      and  fail_when_unscored                  (nothing could be evaluated)

The score and the checks come from the deterministic quality engine; nothing here is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from datasentinel.exceptions import QualityGateError

if TYPE_CHECKING:  # pragma: no cover
    from datasentinel.config import Settings
    from datasentinel.quality.models import QualityReport


@dataclass(frozen=True)
class GateConfig:
    """Thresholds of a quality gate."""

    min_score: float = 60.0
    fail_on_critical: bool = True
    fail_when_unscored: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.min_score, bool) or not 0.0 <= float(self.min_score) <= 100.0:
            raise ValueError("min_score must be between 0 and 100")

    @classmethod
    def from_settings(cls, settings: Optional["Settings"] = None) -> "GateConfig":
        """Gate thresholds from ``DATASENTINEL_QUALITY_GATE_*`` settings."""
        from datasentinel.config import get_settings

        settings = settings or get_settings()
        return cls(settings.quality_gate_min_score, settings.quality_gate_fail_on_critical,
                   settings.quality_gate_fail_when_unscored)


@dataclass(frozen=True)
class CriticalCheck:
    """A failed CRITICAL check that contributed to a gate failure."""

    rule_id: str
    target: str
    affected_pct: float


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quality gate, with every reason it failed (empty when it passed)."""

    passed: bool
    reasons: Tuple[str, ...]
    overall_score: Optional[float]
    grade: str
    critical_failures: Tuple[CriticalCheck, ...]
    config: GateConfig

    def to_context(self) -> dict:
        """JSON-compatible description for a failure's context (metadata only, no data values)."""
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "overall_score": self.overall_score,
            "grade": self.grade,
            "critical_failures": [{"rule_id": c.rule_id, "target": c.target, "affected_pct": c.affected_pct}
                                  for c in self.critical_failures],
            "min_score": self.config.min_score,
            "fail_on_critical": self.config.fail_on_critical,
            "fail_when_unscored": self.config.fail_when_unscored,
        }


def evaluate_quality_gate(report: "QualityReport", config: Optional[GateConfig] = None) -> GateResult:
    """Evaluate the gate against a quality report (pure and deterministic)."""
    # Imported lazily so instrumenting a pipeline does not load the quality engine (and pandas).
    from datasentinel.quality.models import CheckStatus, Severity

    config = config or GateConfig()
    critical = tuple(
        CriticalCheck(r.rule_id, r.target, round(float(r.affected_pct), 4))
        for r in sorted(report.results, key=lambda r: (r.rule_id, r.target))
        if r.status == CheckStatus.FAILED and r.severity == Severity.CRITICAL)
    reasons = []
    if config.fail_on_critical and critical:
        reasons.append(f"{len(critical)} critical quality check(s) failed")
    score = report.overall_score
    if score is not None and score < config.min_score:
        reasons.append(f"quality score {score:.2f} is below the minimum {config.min_score:.2f}")
    if score is None and config.fail_when_unscored:
        reasons.append("no quality checks could be evaluated (unscored)")
    return GateResult(passed=not reasons, reasons=tuple(reasons), overall_score=score, grade=report.grade,
                      critical_failures=critical if config.fail_on_critical else (), config=config)


def enforce_quality_gate(report: "QualityReport", config: Optional[GateConfig] = None) -> GateResult:
    """Evaluate the gate and raise :class:`QualityGateError` (classified DATA_QUALITY) when it fails."""
    result = evaluate_quality_gate(report, config)
    if not result.passed:
        raise QualityGateError("Quality gate failed: " + "; ".join(result.reasons) + ".", gate_result=result)
    return result
