"""Tests for deterministic schema drift detection and baseline loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from datasentinel.exceptions import SchemaContractError
from datasentinel.schema import SchemaContract, SchemaInferrer
from datasentinel.schema.drift import compute_schema_diff, load_baseline


def _schema(df: pd.DataFrame):
    return SchemaInferrer().infer(df)


@pytest.fixture
def current() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": [1, 2, 2, 4],
        "amount": ["10", "x", "12", "13"],         # still numeric (75% < 90% → categorical/text)
        "status": ["a", None, "b", "a"],
        "new_col": [1, 2, 3, 4],
    })


def _changes(diff):
    return {(c.change_type, c.column): c for c in diff.changes}


def test_detects_all_change_types(current) -> None:
    contract = SchemaContract.from_dict({
        "allow_extra_columns": True,
        "columns": {
            "order_id": {"type": "numeric", "unique": True},
            "amount": {"type": "numeric"},
            "status": {"type": "categorical", "nullable": False},
            "legacy": {"type": "text", "required": True},
        },
    })
    diff = compute_schema_diff(contract, _schema(current), current)
    changes = _changes(diff)
    assert changes[("uniqueness_changed", "order_id")].compatibility == "breaking"
    assert changes[("type_changed", "amount")].compatibility == "breaking"
    assert changes[("nullability_changed", "status")].details["null_count"] == 1
    assert changes[("removed", "legacy")].compatibility == "breaking"
    assert changes[("added", "new_col")].compatibility == "non_breaking"
    assert diff.drift_detected and diff.count("breaking") == 4


def test_no_drift_against_own_export(current) -> None:
    schema = _schema(current)
    contract, source = load_baseline(json.dumps(schema.to_dict()))
    assert source == "schema_report"
    assert not compute_schema_diff(contract, schema, current, source).drift_detected


def test_string_family_change_is_non_breaking() -> None:
    df = pd.DataFrame({"name": [f"person {i}" for i in range(20)]})  # text
    contract = SchemaContract.from_dict({"columns": {"name": {"type": "categorical"}}})
    change = compute_schema_diff(contract, _schema(df), df).changes[0]
    assert change.change_type == "type_changed" and change.compatibility == "non_breaking"


def test_rename_candidate() -> None:
    df = pd.DataFrame({"customer_email": ["a", "b", "c"]})
    contract = SchemaContract.from_dict({"columns": {"cust_email": {"type": "text"}}})
    changes = _changes(compute_schema_diff(contract, _schema(df), df))
    assert ("rename_candidate", "customer_email") in changes
    assert changes[("rename_candidate", "customer_email")].before == "cust_email"


def test_extra_columns_disallowed_need_review() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    contract = SchemaContract.from_dict({"allow_extra_columns": False, "columns": {"a": {"type": "numeric"}}})
    assert _changes(compute_schema_diff(contract, _schema(df), df))[("added", "b")].compatibility == "needs_review"


def test_load_baseline_formats_and_errors() -> None:
    contract, source = load_baseline('{"columns": {"a": {"type": "numeric"}}}')
    assert source == "contract" and "a" in contract.columns
    for bad in ("not json", "[]", '{"x": 1}', '{"columns": [{"name": "a", "semantic_type": "weird"}]}'):
        with pytest.raises(SchemaContractError):
            load_baseline(bad)


def test_diff_serialises(current) -> None:
    contract = SchemaContract.from_dict({"columns": {"gone": {"type": "numeric"}}})
    data = json.loads(json.dumps(compute_schema_diff(contract, _schema(current), current).to_dict()))
    assert data["drift_detected"] is True and data["summary"]["breaking"] == 1
