"""Deterministic schema inference.

The inferrer assigns each column a :class:`SemanticType` using the storage dtype first
and, for text columns, the *share of values that successfully parse* as boolean,
numeric or datetime. Thresholds (default 90%) make the result tolerant of a few dirty
values — those values are later reported by the quality engine as *invalid* rather than
silently changing the inferred type of the whole column.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from datasentinel.schema.models import ColumnSchema, PrimaryKeyCandidate, SchemaReport, SemanticType
from datasentinel.utils.parsing import (
    BOOLEAN_TOKENS,
    looks_like_date_column,
    missing_mask,
    parse_boolean,
    parse_datetime,
    parse_numeric,
)

logger = logging.getLogger(__name__)

_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|Id$|ID$|uuid|guid|(^|_)key$|(^|_)code$|_no$|_num(ber)?$)")
_LEADING_ZERO = re.compile(r"^0\d+$")


@dataclass
class InferenceConfig:
    """Tunable thresholds for schema inference."""

    numeric_threshold: float = 0.9
    datetime_threshold: float = 0.9
    datetime_threshold_name_hint: float = 0.6
    boolean_threshold: float = 0.95
    categorical_max_unique: int = 50
    categorical_max_unique_ratio: float = 0.5
    text_min_avg_length: float = 50.0
    detection_sample_size: int = 10_000
    sample_values: int = 5
    max_composite_key_columns: int = 20
    max_composite_key_results: int = 3


class SchemaInferrer:
    """Infer column types, semantic categories and primary-key candidates."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()

    # ------------------------------------------------------------------ public API

    def infer(self, df: pd.DataFrame) -> SchemaReport:
        """Infer a :class:`SchemaReport` for ``df``."""
        columns = [self._infer_column(name, position, df[name]) for position, name in enumerate(df.columns)]
        report = SchemaReport(n_rows=int(len(df)), n_columns=int(df.shape[1]), columns=columns)
        report.primary_key_candidates = self._detect_primary_keys(df, report)
        logger.info(
            "Inferred schema: %s; primary key candidates: %s",
            report.type_summary(),
            [pk.label for pk in report.primary_key_candidates] or "none",
        )
        return report

    def cast(self, df: pd.DataFrame, report: SchemaReport) -> pd.DataFrame:
        """Return a copy of ``df`` with columns coerced to their inferred semantic types.

        Values that cannot be coerced become null; comparing the raw and typed frames is
        how the quality engine identifies invalid values.
        """
        typed = df.copy()
        for column in report.columns:
            series = df[column.name]
            if column.semantic_type == SemanticType.NUMERIC and column.inferred_from_string:
                typed[column.name] = parse_numeric(series)
            elif column.semantic_type == SemanticType.DATETIME:
                typed[column.name] = parse_datetime(series)
            elif column.semantic_type == SemanticType.BOOLEAN:
                typed[column.name] = parse_boolean(series)
            elif column.semantic_type in (SemanticType.CATEGORICAL, SemanticType.TEXT, SemanticType.EMPTY):
                if series.dtype == object:
                    typed[column.name] = series.mask(missing_mask(series))
        return typed

    # ------------------------------------------------------------------ column inference

    def _infer_column(self, name: str, position: int, series: pd.Series) -> ColumnSchema:
        missing = missing_mask(series)
        present = series[~missing]
        null_count = int(missing.sum())
        unique_count = int(present.nunique(dropna=True))
        samples = present.drop_duplicates().head(self.config.sample_values).tolist()
        column = ColumnSchema(
            name=str(name),
            position=position,
            pandas_dtype=str(series.dtype),
            semantic_type=SemanticType.EMPTY,
            storage_type="empty",
            nullable=null_count > 0,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=samples,
        )
        blank_count = int(missing.sum() - series.isna().sum())
        if blank_count:
            column.notes.append(f"{blank_count} whitespace-only value(s) treated as missing")

        if present.empty:
            column.notes.append("Column contains no values")
            return column

        dtype = series.dtype
        if is_bool_dtype(dtype):
            column.semantic_type, column.storage_type = SemanticType.BOOLEAN, "boolean"
        elif is_datetime64_any_dtype(dtype):
            column.semantic_type, column.storage_type = SemanticType.DATETIME, "datetime"
        elif is_numeric_dtype(dtype):
            self._classify_native_numeric(column, present)
        elif isinstance(dtype, pd.CategoricalDtype):
            column.semantic_type, column.storage_type = SemanticType.CATEGORICAL, "category"
        else:
            self._classify_text(column, present)
        return column

    @staticmethod
    def _classify_native_numeric(column: ColumnSchema, present: pd.Series) -> None:
        column.semantic_type = SemanticType.NUMERIC
        if is_integer_dtype(present.dtype):
            column.storage_type = "integer"
        else:
            column.storage_type = "float"
            if is_float_dtype(present.dtype) and (present.dropna() % 1 == 0).all():
                column.notes.append("Whole numbers stored as float (often caused by missing values)")
        distinct = set(present.unique().tolist())
        if distinct and distinct <= {0, 1}:
            column.notes.append("Binary 0/1 values; may represent a boolean flag")

    def _classify_text(self, column: ColumnSchema, present: pd.Series) -> None:
        cfg = self.config
        python_types = present.map(lambda v: type(v).__name__).unique().tolist()
        column.storage_type = "string" if python_types == ["str"] else "mixed"
        if column.storage_type == "mixed":
            column.notes.append(f"Mixed value types: {sorted(python_types)}")

        sample = present
        if len(present) > cfg.detection_sample_size:
            sample = present.sample(cfg.detection_sample_size, random_state=0)
        strings = sample.astype(str).str.strip()
        n = len(strings)

        # Boolean: tokens like yes/no/true/false (pure 0/1 is left numeric).
        lowered = strings.str.lower()
        bool_ratio = float(lowered.isin(BOOLEAN_TOKENS).mean())
        if bool_ratio >= cfg.boolean_threshold and not set(lowered.unique()) <= {"0", "1"}:
            self._mark_parsed(column, SemanticType.BOOLEAN, bool_ratio)
            return

        # Numeric.
        numeric_ratio = float(parse_numeric(strings).notna().mean())
        if numeric_ratio >= cfg.numeric_threshold:
            leading_zero = int(strings.str.match(_LEADING_ZERO).sum())
            if leading_zero:
                column.notes.append(
                    f"{leading_zero} value(s) have leading zeros; kept as text to preserve identifier codes"
                )
            else:
                self._mark_parsed(column, SemanticType.NUMERIC, numeric_ratio)
                return

        # Datetime: only consider values that contain digits, and avoid re-reading pure numbers.
        digit_ratio = float(strings.str.contains(r"\d", regex=True).mean())
        name_hint = looks_like_date_column(column.name)
        dt_threshold = cfg.datetime_threshold_name_hint if name_hint else cfg.datetime_threshold
        if digit_ratio >= dt_threshold and numeric_ratio < cfg.numeric_threshold:
            datetime_ratio = float(parse_datetime(strings).notna().mean())
            if datetime_ratio >= dt_threshold:
                self._mark_parsed(column, SemanticType.DATETIME, datetime_ratio)
                return

        # Categorical vs free text.
        unique_ratio = column.unique_count / max(len(present), 1)
        avg_length = float(strings.str.len().mean()) if n else 0.0
        if avg_length >= cfg.text_min_avg_length:
            column.semantic_type = SemanticType.TEXT
        elif unique_ratio <= cfg.categorical_max_unique_ratio or (
            column.unique_count <= cfg.categorical_max_unique and unique_ratio < 0.9
        ):
            column.semantic_type = SemanticType.CATEGORICAL
        else:
            column.semantic_type = SemanticType.TEXT

    @staticmethod
    def _mark_parsed(column: ColumnSchema, semantic_type: SemanticType, ratio: float) -> None:
        column.semantic_type = semantic_type
        column.inferred_from_string = True
        column.parse_success_ratio = round(ratio, 4)
        if ratio < 1.0:
            column.notes.append(
                f"{(1 - ratio) * 100:.1f}% of values could not be parsed as {semantic_type.value}"
            )

    # ------------------------------------------------------------------ primary keys

    def _detect_primary_keys(self, df: pd.DataFrame, report: SchemaReport) -> List[PrimaryKeyCandidate]:
        n_rows = len(df)
        if n_rows < 2:
            return []
        eligible: List[str] = []
        candidates: List[PrimaryKeyCandidate] = []
        for column in report.columns:
            if column.null_count > 0 or column.semantic_type in (SemanticType.BOOLEAN, SemanticType.EMPTY):
                continue
            series = df[column.name]
            if is_float_dtype(series.dtype) and not (series % 1 == 0).all():
                continue  # continuous measurements are not identifiers
            eligible.append(column.name)
            if column.unique_count == n_rows:
                candidates.append(self._score_key(column))

        if not candidates:
            candidates = self._detect_composite_keys(df, eligible)
        candidates.sort(key=lambda c: (-c.confidence, len(c.columns), c.columns[0]))
        return candidates

    @staticmethod
    def _score_key(column: ColumnSchema) -> PrimaryKeyCandidate:
        confidence, reasons = 0.5, ["All values are unique and non-null"]
        if _ID_NAME_PATTERN.search(column.name):
            confidence += 0.35
            reasons.append("Column name follows an identifier naming convention")
        if column.storage_type == "integer" or (column.semantic_type == SemanticType.CATEGORICAL):
            confidence += 0.1
            reasons.append(f"{column.storage_type.capitalize()} values are typical for keys")
        if column.semantic_type == SemanticType.DATETIME:
            confidence -= 0.25
            reasons.append("Timestamps are unique here but rarely stable keys")
        if column.semantic_type == SemanticType.TEXT and column.position > 0:
            confidence -= 0.1
        if column.position == 0:
            confidence += 0.05
            reasons.append("First column in the dataset")
        return PrimaryKeyCandidate(columns=[column.name], confidence=round(min(max(confidence, 0.05), 0.99), 2), reasons=reasons)

    def _detect_composite_keys(self, df: pd.DataFrame, eligible: List[str]) -> List[PrimaryKeyCandidate]:
        columns = eligible[: self.config.max_composite_key_columns]
        found: List[PrimaryKeyCandidate] = []
        pairs: List[Tuple[str, str]] = list(itertools.combinations(columns, 2))
        for pair in pairs:
            if not df.duplicated(subset=list(pair)).any():
                found.append(
                    PrimaryKeyCandidate(
                        columns=list(pair),
                        confidence=0.45 if any(_ID_NAME_PATTERN.search(c) for c in pair) else 0.3,
                        reasons=["No single column is unique, but this combination uniquely identifies every row"],
                    )
                )
                if len(found) >= self.config.max_composite_key_results:
                    break
        return found
