"""Statistical profiling of datasets.

The profiler consumes both the *raw* DataFrame (for row-level facts such as duplicates
and missing values exactly as delivered) and the *typed* DataFrame produced by
:meth:`SchemaInferrer.cast` (for type-aware statistics such as numeric moments).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from datasentinel.profiling.models import (
    BooleanStats,
    CategoricalStats,
    ColumnProfile,
    DatasetProfile,
    DatetimeStats,
    NumericStats,
    ValueFrequency,
)
from datasentinel.schema.models import ColumnSchema, SchemaReport, SemanticType

logger = logging.getLogger(__name__)


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) or math.isinf(result) else result


class DataProfiler:
    """Compute dataset- and column-level statistics.

    Args:
        histogram_bins: Maximum number of histogram bins for numeric columns.
        top_n: Number of most frequent values to report for categorical columns.
    """

    def __init__(self, histogram_bins: int = 20, top_n: int = 10) -> None:
        self.histogram_bins = histogram_bins
        self.top_n = top_n

    def profile(self, raw: pd.DataFrame, typed: pd.DataFrame, schema: SchemaReport) -> DatasetProfile:
        """Profile a dataset.

        Args:
            raw: The DataFrame exactly as ingested.
            typed: The DataFrame coerced to inferred types.
            schema: The inferred schema for ``raw``.
        """
        n_rows, n_columns = raw.shape
        total_cells = int(n_rows * n_columns)
        missing_cells = int(sum(c.null_count for c in schema.columns))
        duplicate_rows = int(raw.duplicated().sum())

        profile = DatasetProfile(
            n_rows=int(n_rows),
            n_columns=int(n_columns),
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_cells_pct=(missing_cells / total_cells * 100) if total_cells else 0.0,
            duplicate_rows=duplicate_rows,
            duplicate_rows_pct=(duplicate_rows / n_rows * 100) if n_rows else 0.0,
            memory_bytes=int(raw.memory_usage(deep=True).sum()),
        )
        for column in schema.columns:
            profile.columns[column.name] = self._profile_column(typed[column.name], column, n_rows)
        logger.info(
            "Profiled %d columns: %.2f%% missing cells, %d duplicate rows",
            n_columns, profile.missing_cells_pct, duplicate_rows,
        )
        return profile

    # ------------------------------------------------------------------ columns

    def _profile_column(self, series: pd.Series, schema: ColumnSchema, n_rows: int) -> ColumnProfile:
        non_null = series.dropna()
        non_null_count = n_rows - schema.null_count
        profile = ColumnProfile(
            name=schema.name,
            semantic_type=schema.semantic_type,
            dtype=str(series.dtype),
            non_null_count=int(non_null_count),
            null_count=schema.null_count,
            null_pct=(schema.null_count / n_rows * 100) if n_rows else 0.0,
            unique_count=schema.unique_count,
            unique_pct=(schema.unique_count / non_null_count * 100) if non_null_count else 0.0,
            is_unique=non_null_count > 0 and schema.unique_count == non_null_count,
            is_constant=schema.unique_count == 1,
        )
        stype = schema.semantic_type
        if stype == SemanticType.NUMERIC and not non_null.empty:
            profile.numeric = self._numeric_stats(non_null)
        elif stype == SemanticType.DATETIME and not non_null.empty:
            profile.datetime = self._datetime_stats(non_null)
        elif stype == SemanticType.BOOLEAN and not non_null.empty:
            profile.boolean = self._boolean_stats(non_null)
        elif stype in (SemanticType.CATEGORICAL, SemanticType.TEXT) and not non_null.empty:
            profile.categorical = self._categorical_stats(non_null)
        return profile

    def _numeric_stats(self, values: pd.Series) -> Optional[NumericStats]:
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        finite_mask = np.isfinite(numeric.to_numpy())
        infinite = int((~finite_mask).sum())
        numeric = numeric[finite_mask]
        if numeric.empty:
            return None
        q = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        unique = numeric.nunique()
        bins = int(max(1, min(self.histogram_bins, unique)))
        counts, edges = np.histogram(numeric.to_numpy(), bins=bins)
        return NumericStats(
            count=int(numeric.size),
            min=float(numeric.min()),
            max=float(numeric.max()),
            mean=float(numeric.mean()),
            median=float(q.loc[0.5]),
            std=_safe_float(numeric.std()),
            q1=float(q.loc[0.25]),
            q3=float(q.loc[0.75]),
            iqr=float(q.loc[0.75] - q.loc[0.25]),
            p05=float(q.loc[0.05]),
            p95=float(q.loc[0.95]),
            skewness=_safe_float(numeric.skew()) if numeric.size > 2 else None,
            kurtosis=_safe_float(numeric.kurt()) if numeric.size > 3 else None,
            zeros=int((numeric == 0).sum()),
            negatives=int((numeric < 0).sum()),
            infinite=infinite,
            histogram_counts=[int(c) for c in counts],
            histogram_edges=[float(e) for e in edges],
        )

    def _categorical_stats(self, values: pd.Series) -> CategoricalStats:
        counts = values.astype(str).value_counts()
        total = int(counts.sum())
        probabilities = counts / total
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
        top = counts.head(self.top_n)
        lengths = values.astype(str).str.len()
        return CategoricalStats(
            cardinality=int(len(counts)),
            mode=counts.index[0],
            mode_pct=float(counts.iloc[0] / total * 100),
            entropy=round(entropy, 4),
            normalized_entropy=round(entropy / max_entropy, 4) if max_entropy else 0.0,
            top_values=[ValueFrequency(value=k, count=int(v), pct=float(v / total * 100)) for k, v in top.items()],
            other_values_count=int(total - top.sum()),
            min_length=int(lengths.min()),
            mean_length=round(float(lengths.mean()), 2),
            max_length=int(lengths.max()),
        )

    @staticmethod
    def _datetime_stats(values: pd.Series) -> DatetimeStats:
        minimum, maximum = values.min(), values.max()
        return DatetimeStats(
            min=minimum,
            max=maximum,
            # Python datetimes avoid pandas' ±292-year Timedelta overflow on very wide ranges.
            span_days=round((maximum.to_pydatetime() - minimum.to_pydatetime()).total_seconds() / 86400, 2),
            distinct_dates=int(values.dt.normalize().nunique()),
        )

    @staticmethod
    def _boolean_stats(values: pd.Series) -> BooleanStats:
        as_bool = values.astype(bool)
        true_count = int(as_bool.sum())
        total = int(as_bool.size)
        return BooleanStats(
            true_count=true_count,
            false_count=total - true_count,
            true_pct=true_count / total * 100 if total else 0.0,
        )
