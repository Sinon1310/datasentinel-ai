"""Transparent quality scoring.

Method
------
1. **Check score.** Every evaluated (non-skipped) check scores
   ``100 × (1 − penalty[severity])``. Passed checks score 100. Default penalties:
   low 0.25, medium 0.5, high 0.75, critical 1.0.
2. **Dimension score.** The mean of its check scores.
3. **Overall score.** The weighted mean of dimension scores. Dimensions with no
   evaluated checks are excluded and the remaining weights are re-normalised, so a
   dataset with no numeric columns is not rewarded or punished for "accuracy".
4. **Grade.** A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, otherwise F.

Each step is recorded in the report (``effective_weight`` and ``contribution`` per
dimension; ``score`` per check) so the final number can be reproduced by hand.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from datasentinel.quality.models import CheckStatus, DimensionScore, QualityDimension, RuleResult, Severity

_GRADES = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"))


def check_score(result: RuleResult, penalties: Dict[Severity, float]) -> Optional[float]:
    """Score a single check in [0, 100]; ``None`` for skipped checks."""
    if result.status == CheckStatus.SKIPPED:
        return None
    if result.status == CheckStatus.PASSED:
        return 100.0
    return round(100.0 * (1.0 - penalties.get(result.severity, 1.0)), 4)


def grade_for(score: Optional[float]) -> str:
    """Letter grade for an overall score."""
    if score is None:
        return "N/A"
    for threshold, grade in _GRADES:
        if score >= threshold:
            return grade
    return "F"


def compute_scores(
    results: List[RuleResult],
    weights: Dict[QualityDimension, float],
    penalties: Dict[Severity, float],
) -> Tuple[Optional[float], List[DimensionScore]]:
    """Score every check (in place) and aggregate into dimension and overall scores.

    Returns:
        ``(overall_score, dimension_scores)``; ``overall_score`` is ``None`` when no
        check could be evaluated.
    """
    by_dimension: Dict[QualityDimension, List[RuleResult]] = {d: [] for d in QualityDimension}
    for result in results:
        result.score = check_score(result, penalties)
        by_dimension[result.dimension].append(result)

    raw_scores: Dict[QualityDimension, Optional[float]] = {}
    for dimension, dim_results in by_dimension.items():
        scores = [r.score for r in dim_results if r.score is not None]
        raw_scores[dimension] = sum(scores) / len(scores) if scores else None

    active_weight = sum(weights.get(d, 0.0) for d, s in raw_scores.items() if s is not None)
    dimension_scores: List[DimensionScore] = []
    overall = 0.0
    for dimension in QualityDimension:
        score = raw_scores[dimension]
        weight = weights.get(dimension, 0.0)
        effective = (weight / active_weight) if (score is not None and active_weight) else 0.0
        contribution = (score or 0.0) * effective
        overall += contribution
        evaluated = [r for r in by_dimension[dimension] if r.score is not None]
        dimension_scores.append(
            DimensionScore(
                dimension=dimension,
                weight=weight,
                effective_weight=round(effective, 4),
                score=None if score is None else round(score, 2),
                checks_evaluated=len(evaluated),
                checks_failed=sum(1 for r in evaluated if r.status == CheckStatus.FAILED),
                contribution=round(contribution, 2),
            )
        )
    if not active_weight:
        return None, dimension_scores
    return round(overall, 2), dimension_scores
