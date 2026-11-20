"""Data models for dataset and column profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from datasentinel.schema.models import SemanticType
from datasentinel.utils.serialization import to_builtin


@dataclass
class NumericStats:
    """Descriptive statistics for a numeric column (finite values only)."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    std: Optional[float]
    q1: float
    q3: float
    iqr: float
    p05: float
    p95: float
    skewness: Optional[float]
    kurtosis: Optional[float]
    zeros: int
    negatives: int
    infinite: int
    histogram_counts: List[int] = field(default_factory=list)
    histogram_edges: List[float] = field(default_factory=list)


@dataclass
class ValueFrequency:
    """Frequency of a single value."""

    value: Any
    count: int
    pct: float


@dataclass
class CategoricalStats:
    """Distribution statistics for categorical / text columns."""

    cardinality: int
    mode: Any
    mode_pct: float
    entropy: Optional[float]
    normalized_entropy: Optional[float]
    top_values: List[ValueFrequency] = field(default_factory=list)
    other_values_count: int = 0
    min_length: Optional[int] = None
    mean_length: Optional[float] = None
    max_length: Optional[int] = None


@dataclass
class DatetimeStats:
    """Range statistics for datetime columns."""

    min: Any
    max: Any
    span_days: float
    distinct_dates: int


@dataclass
class BooleanStats:
    """Counts for boolean columns."""

    true_count: int
    false_count: int
    true_pct: float


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    semantic_type: SemanticType
    dtype: str
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    is_unique: bool
    is_constant: bool
    numeric: Optional[NumericStats] = None
    categorical: Optional[CategoricalStats] = None
    datetime: Optional[DatetimeStats] = None
    boolean: Optional[BooleanStats] = None


@dataclass
class DatasetProfile:
    """Dataset-level profile plus per-column profiles."""

    n_rows: int
    n_columns: int
    total_cells: int
    missing_cells: int
    missing_cells_pct: float
    duplicate_rows: int
    duplicate_rows_pct: float
    memory_bytes: int
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_builtin(self)

    def summary_dataframe(self) -> pd.DataFrame:
        """One row per column with headline statistics."""
        rows = []
        for p in self.columns.values():
            row: Dict[str, Any] = {
                "column": p.name,
                "semantic_type": p.semantic_type.value,
                "non_null": p.non_null_count,
                "null_count": p.null_count,
                "null_pct": round(p.null_pct, 2),
                "unique_count": p.unique_count,
                "unique_pct": round(p.unique_pct, 2),
                "min": None, "max": None, "mean": None, "median": None, "std": None, "cardinality": None,
            }
            if p.numeric:
                # min/max are text so numeric and datetime ranges share one display column.
                row.update(min=f"{p.numeric.min:.6g}", max=f"{p.numeric.max:.6g}", mean=p.numeric.mean,
                           median=p.numeric.median, std=p.numeric.std)
            elif p.datetime:
                row.update(min=str(p.datetime.min), max=str(p.datetime.max))
            if p.categorical:
                row["cardinality"] = p.categorical.cardinality
            rows.append(row)
        return pd.DataFrame(rows)
