"""Deterministic value parsers shared by schema inference and quality validation.

Keeping coercion logic in one place guarantees that the type the schema engine
*infers* is exactly the type the quality engine *validates against*.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0"})
BOOLEAN_TOKENS = TRUE_TOKENS | FALSE_TOKENS

_DATE_NAME_TOKENS = frozenset(
    {"date", "time", "timestamp", "datetime", "dt", "at", "dob", "birthday", "created", "updated", "modified", "ts"}
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def blank_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask of values that are strings containing only whitespace."""
    if is_numeric_dtype(series.dtype) or is_datetime64_any_dtype(series.dtype) or is_bool_dtype(series.dtype):
        return pd.Series(False, index=series.index)
    return series.map(lambda v: isinstance(v, str) and v.strip() == "").astype(bool)


def missing_mask(series: pd.Series) -> pd.Series:
    """Return a mask of values considered *missing*: nulls plus whitespace-only strings."""
    return series.isna() | blank_mask(series)


def looks_like_date_column(name: str) -> bool:
    """Heuristic: does the column name suggest it holds dates or timestamps?

    Names are tokenised on separators and camelCase boundaries, so ``order_date``,
    ``createdAt`` and ``event_ts`` match while ``candidate`` or ``validated`` do not.
    """
    tokens = _TOKEN_SPLIT.split(_CAMEL_BOUNDARY.sub("_", str(name)))
    return any(token.lower() in _DATE_NAME_TOKENS for token in tokens if token)


def parse_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric; unparseable values become ``NaN``."""
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        return series
    present = series[~missing_mask(series)]
    parsed = pd.to_numeric(present.astype(str).str.strip(), errors="coerce")
    return parsed.reindex(series.index)


def parse_boolean(series: pd.Series) -> pd.Series:
    """Coerce a series to pandas' nullable ``boolean`` dtype; unknown tokens become ``<NA>``."""
    if is_bool_dtype(series.dtype):
        return series.astype("boolean")
    mapping = {token: True for token in TRUE_TOKENS}
    mapping.update({token: False for token in FALSE_TOKENS})
    present = series[~missing_mask(series)]
    parsed = present.astype(str).str.strip().str.lower().map(mapping)
    return parsed.reindex(series.index).astype("boolean")


def _normalize_timezone(parsed: pd.Series) -> pd.Series:
    """Convert tz-aware or mixed-offset results to naive UTC ``datetime64[ns]``."""
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    if not is_datetime64_any_dtype(parsed.dtype):
        # Mixed UTC offsets produce an object series; normalise through UTC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(parsed, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)
    return parsed


def _to_datetime(values: pd.Series, mixed: bool) -> pd.Series:
    kwargs = {"errors": "coerce"}
    if mixed:
        kwargs["format"] = "mixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(values, **kwargs)
        except (ValueError, TypeError, OverflowError):
            parsed = pd.to_datetime(values, utc=True, **kwargs)
    return _normalize_timezone(parsed)


def parse_datetime(series: pd.Series) -> pd.Series:
    """Coerce a series to naive (UTC) ``datetime64[ns]``; unparseable values become ``NaT``.

    A fast vectorised pass (single inferred format) runs first; only values it could
    not parse are retried with per-element ``format="mixed"`` parsing. Dates outside
    pandas' nanosecond range (e.g. ``9999-12-31``) are treated as unparseable.
    """
    if is_datetime64_any_dtype(series.dtype):
        return _normalize_timezone(series)
    present = series[~missing_mask(series)]
    if present.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    strings = present.astype(str).str.strip()
    parsed = _to_datetime(strings, mixed=False)
    failed = parsed.isna()
    if failed.any():
        retry = _to_datetime(strings[failed], mixed=True)
        parsed = parsed.astype("datetime64[ns]")
        parsed[failed] = retry.astype("datetime64[ns]")
    return parsed.astype("datetime64[ns]").reindex(series.index)
