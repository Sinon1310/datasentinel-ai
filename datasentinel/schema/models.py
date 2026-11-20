"""Data models for inferred schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from datasentinel.utils.serialization import to_builtin


class SemanticType(str, Enum):
    """Logical (semantic) type of a column, independent of its storage dtype."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMPTY = "empty"


@dataclass
class ColumnSchema:
    """Inferred schema for a single column.

    Attributes:
        name: Column name.
        position: Zero-based column position.
        pandas_dtype: Storage dtype as loaded (e.g. ``object``, ``int64``).
        semantic_type: Inferred logical type.
        storage_type: Physical representation (integer, float, string, boolean, datetime, mixed, empty).
        nullable: Whether any value is missing.
        null_count: Missing values (nulls plus whitespace-only strings).
        unique_count: Distinct non-missing values.
        inferred_from_string: True when a typed semantic (numeric/date/boolean) was parsed from text.
        parse_success_ratio: Share of non-missing values that parsed into the semantic type.
        sample_values: A few distinct example values.
        notes: Human-readable observations made during inference.
    """

    name: str
    position: int
    pandas_dtype: str
    semantic_type: SemanticType
    storage_type: str
    nullable: bool
    null_count: int
    unique_count: int
    inferred_from_string: bool = False
    parse_success_ratio: Optional[float] = None
    sample_values: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class PrimaryKeyCandidate:
    """A column (or column combination) that uniquely identifies every row."""

    columns: List[str]
    confidence: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        """True when the key spans more than one column."""
        return len(self.columns) > 1

    @property
    def label(self) -> str:
        """Display label, e.g. ``order_id`` or ``store_id + sku``."""
        return " + ".join(self.columns)


@dataclass
class SchemaReport:
    """Structured schema report for a dataset."""

    n_rows: int
    n_columns: int
    columns: List[ColumnSchema]
    primary_key_candidates: List[PrimaryKeyCandidate] = field(default_factory=list)

    def column(self, name: str) -> ColumnSchema:
        """Return the schema of the named column.

        Raises:
            KeyError: If the column does not exist.
        """
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def columns_of_type(self, *types: SemanticType) -> List[str]:
        """Names of columns whose semantic type is one of ``types``."""
        return [c.name for c in self.columns if c.semantic_type in types]

    @property
    def primary_key(self) -> Optional[PrimaryKeyCandidate]:
        """The highest-confidence primary-key candidate, if any."""
        return self.primary_key_candidates[0] if self.primary_key_candidates else None

    def type_summary(self) -> Dict[str, int]:
        """Count of columns per semantic type."""
        summary: Dict[str, int] = {}
        for column in self.columns:
            summary[column.semantic_type.value] = summary.get(column.semantic_type.value, 0) + 1
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary."""
        data = to_builtin(self)
        data["type_summary"] = self.type_summary()
        return data

    def to_dataframe(self) -> pd.DataFrame:
        """Tabular view of the column schemas, suitable for display."""
        key_columns = {c for pk in self.primary_key_candidates if not pk.is_composite for c in pk.columns}
        rows = [
            {
                "column": c.name,
                "semantic_type": c.semantic_type.value,
                "storage_type": c.storage_type,
                "pandas_dtype": c.pandas_dtype,
                "nullable": c.nullable,
                "null_count": c.null_count,
                "unique_count": c.unique_count,
                "pk_candidate": c.name in key_columns,
                "parse_success_ratio": c.parse_success_ratio,
                "sample_values": ", ".join(str(v) for v in c.sample_values),
                "notes": "; ".join(c.notes),
            }
            for c in self.columns
        ]
        return pd.DataFrame(rows)
