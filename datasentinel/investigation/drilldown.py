"""Deterministic drill-down queries over an :class:`AnalysisResult`.

These answer *conditional* questions the profiler does not (where / when / with what do
issues occur). They reuse the engine's own definitions so the answers are consistent with
the quality report:

* ``missing``       — :func:`missing_mask` (nulls + blank strings), as in ``MissingValuesRule``
* ``invalid_type``  — present in raw but null in the typed frame, as in ``InvalidDataTypeRule``
* ``invalid_date``  — unparseable or outside the range recorded by ``InvalidDateRule``
* ``outlier``       — outside the fences recorded by ``OutlierIQRRule`` (not recomputed)
* ``duplicate``     — duplicated rows (or duplicated values when a column is given)

All outputs are privacy-filtered through :class:`PrivacyGuard`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from datasentinel.exceptions import DataSentinelError
from datasentinel.investigation.privacy import PrivacyGuard, safe_identifier
from datasentinel.pipeline import AnalysisResult
from datasentinel.quality.models import RuleResult
from datasentinel.schema.models import SemanticType
from datasentinel.utils.parsing import missing_mask, parse_datetime

ISSUE_TYPES = ("missing", "invalid_type", "invalid_date", "outlier", "duplicate")
_MAX_SEGMENTS_CARDINALITY = 100
_FREQUENCIES = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y"}


class DrilldownError(DataSentinelError):
    """A drill-down query was invalid for this dataset."""


def _require_column(result: AnalysisResult, column: str) -> None:
    if column not in result.raw_dataframe.columns:
        raise DrilldownError(f"Unknown column '{column}'. Available: {list(result.raw_dataframe.columns)}")


def _rule_result(result: AnalysisResult, rule_id: str, column: str) -> Optional[RuleResult]:
    return next((r for r in result.quality.results if r.rule_id == rule_id and r.column == column), None)


def issue_mask(result: AnalysisResult, column: Optional[str], issue: str) -> pd.Series:
    """Boolean mask of rows exhibiting ``issue`` (for ``column`` where applicable)."""
    if issue not in ISSUE_TYPES:
        raise DrilldownError(f"Unknown issue '{issue}'. Expected one of {ISSUE_TYPES}.")
    raw, typed = result.raw_dataframe, result.typed_dataframe

    if issue == "duplicate":
        if column is None:
            return raw.duplicated(keep=False)
        _require_column(result, column)
        present = ~missing_mask(raw[column])
        return raw[column].duplicated(keep=False) & present

    if column is None:
        raise DrilldownError(f"Issue '{issue}' requires a column.")
    _require_column(result, column)
    stype = result.schema.column(column).semantic_type
    present = ~missing_mask(raw[column])

    if issue == "missing":
        return ~present
    if issue == "invalid_type":
        if stype not in (SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.DATETIME):
            raise DrilldownError(f"'invalid_type' applies to numeric, boolean or datetime columns; '{column}' is {stype.value}.")
        return present & typed[column].isna()
    if issue == "invalid_date":
        rule = _rule_result(result, "validity.dates", column)
        if rule is None or "valid_range" not in rule.details:
            raise DrilldownError(f"Column '{column}' was not evaluated as a date column.")
        low, high = (pd.Timestamp(v) for v in rule.details["valid_range"])
        parsed = typed[column] if is_datetime64_any_dtype(typed[column].dtype) else parse_datetime(raw[column])
        return present & (parsed.isna() | (parsed < low) | (parsed > high)).fillna(True)
    # outlier
    rule = _rule_result(result, "accuracy.outliers_iqr", column)
    if rule is None or rule.details.get("lower_fence") is None:
        raise DrilldownError(f"No IQR fences were computed for '{column}' (not numeric, a key, or skipped).")
    values = pd.to_numeric(typed[column], errors="coerce")
    return ((values < rule.details["lower_fence"]) | (values > rule.details["upper_fence"])).fillna(False)


def _rate(count: int, total: int) -> float:
    return round(count / total * 100, 4) if total else 0.0


def sample_issue_rows(
    result: AnalysisResult, guard: PrivacyGuard, column: Optional[str], issue: str, limit: int = 10
) -> Dict[str, Any]:
    """Row indices (and privacy-filtered values) of rows exhibiting ``issue``."""
    limit = max(1, min(int(limit), 20))
    mask = issue_mask(result, column, issue)
    rows = mask[mask].index[:limit].tolist()
    payload: Dict[str, Any] = {
        "column": safe_identifier(column) if column else None,
        "issue": issue,
        "matching_rows": int(mask.sum()),
        "matching_pct": _rate(int(mask.sum()), len(mask)),
        "row_indices": [int(i) for i in rows],
        "privacy_mode": guard.mode,
    }
    if column is not None:
        payload["values"] = guard.describe_values(result.raw_dataframe.loc[rows, column].tolist())
    return payload


def issue_rate_by_segment(
    result: AnalysisResult, guard: PrivacyGuard, column: Optional[str], issue: str, by_column: str, top_n: int = 10
) -> Dict[str, Any]:
    """Issue rate per value of ``by_column`` and the lift versus the overall rate."""
    _require_column(result, by_column)
    raw = result.raw_dataframe
    if raw[by_column].nunique(dropna=True) > _MAX_SEGMENTS_CARDINALITY:
        raise DrilldownError(f"'{by_column}' has more than {_MAX_SEGMENTS_CARDINALITY} distinct values; choose a lower-cardinality column.")
    mask = issue_mask(result, column, issue)
    overall = _rate(int(mask.sum()), len(mask))
    segment_values = raw[by_column].mask(missing_mask(raw[by_column]))
    frame = pd.DataFrame({"segment": segment_values.astype(object), "issue": mask.astype(int)})
    frame["segment"] = frame["segment"].where(frame["segment"].notna(), None)
    grouped = frame.groupby("segment", dropna=False)["issue"].agg(["count", "sum"]).reset_index()
    grouped = grouped.sort_values(["sum", "count"], ascending=False).head(max(1, min(int(top_n), 25)))
    segments: List[Dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        rate = _rate(int(row.sum), int(row.count))
        segments.append({
            "segment": guard.label(by_column, row.segment),
            "rows": int(row.count),
            "issue_rows": int(row.sum),
            "issue_rate_pct": rate,
            "lift_vs_overall": round(rate / overall, 3) if overall else None,
        })
    return {
        "column": safe_identifier(column) if column else None,
        "issue": issue,
        "by_column": safe_identifier(by_column),
        "overall_issue_rows": int(mask.sum()),
        "overall_issue_rate_pct": overall,
        "segments": segments,
    }


def issue_trend(
    result: AnalysisResult, column: Optional[str], issue: str, date_column: str, freq: str = "M"
) -> Dict[str, Any]:
    """Issue counts and rates per time period of ``date_column``."""
    _require_column(result, date_column)
    if result.schema.column(date_column).semantic_type != SemanticType.DATETIME:
        raise DrilldownError(f"'{date_column}' is not a datetime column.")
    if freq not in _FREQUENCIES:
        raise DrilldownError(f"Unknown frequency '{freq}'. Expected one of {sorted(_FREQUENCIES)}.")
    mask = issue_mask(result, column, issue)
    dates = result.typed_dataframe[date_column]
    frame = pd.DataFrame({"period": dates.dt.to_period(_FREQUENCIES[freq]).astype(str), "issue": mask.astype(int)})
    frame = frame[dates.notna()]
    grouped = frame.groupby("period")["issue"].agg(["count", "sum"]).reset_index().sort_values("period")
    periods = [
        {"period": str(r.period), "rows": int(r.count), "issue_rows": int(r.sum), "issue_rate_pct": _rate(int(r.sum), int(r.count))}
        for r in grouped.itertuples(index=False)
    ]
    return {
        "column": safe_identifier(column) if column else None,
        "issue": issue,
        "date_column": safe_identifier(date_column),
        "frequency": freq,
        "rows_without_valid_date": int(dates.isna().sum()),
        "periods": periods[-60:],
    }


def missingness_cooccurrence(result: AnalysisResult, columns: List[str]) -> Dict[str, Any]:
    """For each column pair: rows where both are missing and P(B missing | A missing)."""
    if not 2 <= len(columns) <= 8:
        raise DrilldownError("Provide between 2 and 8 columns.")
    for column in columns:
        _require_column(result, column)
    raw = result.raw_dataframe
    masks = {c: missing_mask(raw[c]) for c in columns}
    pairs = []
    for a, b in combinations(columns, 2):
        a_count, b_count = int(masks[a].sum()), int(masks[b].sum())
        both = int((masks[a] & masks[b]).sum())
        pairs.append({
            "column_a": safe_identifier(a), "column_b": safe_identifier(b),
            "missing_a": a_count, "missing_b": b_count, "missing_both": both,
            "p_b_missing_given_a_pct": _rate(both, a_count),
            "p_a_missing_given_b_pct": _rate(both, b_count),
        })
    return {"pairs": pairs}
