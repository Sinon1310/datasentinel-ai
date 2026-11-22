"""Tests for statistical profiling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datasentinel.profiling import DataProfiler
from datasentinel.schema import SchemaInferrer


def _profile(df: pd.DataFrame):
    inferrer = SchemaInferrer()
    schema = inferrer.infer(df)
    return DataProfiler().profile(df, inferrer.cast(df, schema), schema)


def test_dataset_level_counts() -> None:
    df = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"]})
    profile = _profile(df)
    assert profile.n_rows == 4 and profile.n_columns == 2
    assert profile.total_cells == 8
    assert profile.missing_cells == 1
    assert profile.missing_cells_pct == pytest.approx(12.5)
    assert profile.duplicate_rows == 1
    assert profile.duplicate_rows_pct == pytest.approx(25.0)


def test_null_detection_per_column() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0, None], "b": ["x", "", " ", "y"]})
    profile = _profile(df)
    assert profile.columns["a"].null_count == 2
    assert profile.columns["a"].null_pct == pytest.approx(50.0)
    # Empty CSV-style strings and whitespace both count as missing.
    assert profile.columns["b"].null_count == 2


def test_numeric_statistics() -> None:
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
    stats = _profile(df).columns["v"].numeric
    assert stats is not None
    assert stats.min == 1 and stats.max == 100
    assert stats.mean == pytest.approx(14.5)
    assert stats.median == pytest.approx(5.5)
    assert stats.q1 == pytest.approx(3.25) and stats.q3 == pytest.approx(7.75)
    assert stats.iqr == pytest.approx(4.5)
    assert stats.skewness > 0
    assert sum(stats.histogram_counts) == 10
    assert len(stats.histogram_edges) == len(stats.histogram_counts) + 1


def test_numeric_stats_ignore_invalid_strings() -> None:
    df = pd.DataFrame({"v": [str(i) for i in range(1, 20)] + ["bad"]})
    stats = _profile(df).columns["v"].numeric
    assert stats.count == 19
    assert stats.max == 19


def test_categorical_cardinality_and_top_values() -> None:
    df = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 3 + ["c"]})
    cat = _profile(df).columns["c"].categorical
    assert cat.cardinality == 3
    assert cat.mode == "a" and cat.mode_pct == pytest.approx(60.0)
    assert [v.value for v in cat.top_values] == ["a", "b", "c"]
    assert 0 < cat.normalized_entropy < 1


def test_blank_strings_excluded_from_categorical_stats() -> None:
    # Regression: pandas >= 3 stores text as the "str" dtype; blanks must still be treated as missing.
    df = pd.DataFrame({"region": ["N", "S", "   ", "N", "E", "S", "N", "  "]})
    cat = _profile(df).columns["region"].categorical
    assert cat.cardinality == 3
    assert "   " not in [v.value for v in cat.top_values]


def test_unique_and_constant_flags() -> None:
    profile = _profile(pd.DataFrame({"id": [1, 2, 3], "k": ["x", "x", "x"]}))
    assert profile.columns["id"].is_unique and not profile.columns["id"].is_constant
    assert profile.columns["k"].is_constant


def test_datetime_and_boolean_stats() -> None:
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-11", "2024-01-06"], "f": ["yes", "no", "yes"]})
    profile = _profile(df)
    assert profile.columns["d"].datetime.span_days == pytest.approx(10.0)
    assert profile.columns["f"].boolean.true_count == 2


def test_profile_is_json_serializable() -> None:
    df = pd.DataFrame({"v": [1.0, np.nan, np.inf, 4.0], "d": ["2024-01-01", None, "2024-02-01", "2024-03-01"]})
    json.dumps(_profile(df).to_dict())
