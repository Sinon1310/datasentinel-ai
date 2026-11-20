"""Tests for schema inference, primary-key detection and schema contracts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datasentinel.exceptions import SchemaContractError
from datasentinel.ingestion import DatasetLoader
from datasentinel.schema import SchemaContract, SchemaInferrer, SemanticType
from datasentinel.utils.parsing import looks_like_date_column
from tests.conftest import CLEAN_CSV


@pytest.fixture
def inferred():
    df = DatasetLoader().load_bytes(CLEAN_CSV.encode(), "clean.csv").dataframe
    return df, SchemaInferrer().infer(df)


class TestTypeDetection:
    def test_semantic_types_from_csv(self, inferred) -> None:
        _, schema = inferred
        types = {c.name: c.semantic_type for c in schema.columns}
        assert types == {
            "order_id": SemanticType.NUMERIC,
            "customer": SemanticType.TEXT,
            "amount": SemanticType.NUMERIC,
            "order_date": SemanticType.DATETIME,
            "is_active": SemanticType.BOOLEAN,
            "category": SemanticType.CATEGORICAL,
        }
        assert schema.column("order_id").storage_type == "integer"
        assert schema.column("amount").storage_type == "float"
        assert schema.column("order_date").inferred_from_string is True

    def test_native_dtypes(self) -> None:
        df = pd.DataFrame({
            "flag": [True, False, True],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "cat": pd.Categorical(["a", "b", "a"]),
            "empty": [np.nan, np.nan, np.nan],
        })
        schema = SchemaInferrer().infer(df)
        assert schema.column("flag").semantic_type == SemanticType.BOOLEAN
        assert schema.column("ts").semantic_type == SemanticType.DATETIME
        assert schema.column("cat").semantic_type == SemanticType.CATEGORICAL
        assert schema.column("empty").semantic_type == SemanticType.EMPTY

    def test_numeric_strings_tolerate_dirty_values(self) -> None:
        values = [str(i) for i in range(1, 20)] + ["oops"]
        schema = SchemaInferrer().infer(pd.DataFrame({"qty": values}))
        column = schema.column("qty")
        assert column.semantic_type == SemanticType.NUMERIC
        assert column.parse_success_ratio == pytest.approx(0.95)
        assert column.notes  # explains the unparseable share

    def test_leading_zero_codes_stay_text(self) -> None:
        schema = SchemaInferrer().infer(pd.DataFrame({"zip": ["01234", "02345", "10001", "00501"]}))
        assert schema.column("zip").semantic_type != SemanticType.NUMERIC

    def test_binary_01_stays_numeric_with_note(self) -> None:
        schema = SchemaInferrer().infer(pd.DataFrame({"flag": [0, 1, 1, 0, 1]}))
        column = schema.column("flag")
        assert column.semantic_type == SemanticType.NUMERIC
        assert any("boolean" in n for n in column.notes)

    def test_blank_strings_count_as_missing(self) -> None:
        schema = SchemaInferrer().infer(pd.DataFrame({"city": ["NY", "  ", "LA", None, "NY"]}))
        assert schema.column("city").null_count == 2
        assert schema.column("city").nullable is True

    def test_date_name_hint_lowers_threshold(self) -> None:
        values = ["2024-01-01", "2024-01-02", "2024-01-03", "bad", "2024-01-05", "2024-01-06", "x", "2024-01-08"]
        assert SchemaInferrer().infer(pd.DataFrame({"created_at": values})).column("created_at").semantic_type \
            == SemanticType.DATETIME
        assert SchemaInferrer().infer(pd.DataFrame({"label": values})).column("label").semantic_type \
            != SemanticType.DATETIME

    def test_date_name_heuristic(self) -> None:
        assert looks_like_date_column("order_date")
        assert looks_like_date_column("createdAt")
        assert looks_like_date_column("event_ts")
        assert not looks_like_date_column("candidate")
        assert not looks_like_date_column("validated")

    def test_cast_coerces_and_nulls_invalid(self) -> None:
        df = pd.DataFrame({"qty": [str(i) for i in range(1, 20)] + ["oops"]})
        inferrer = SchemaInferrer()
        typed = inferrer.cast(df, inferrer.infer(df))
        assert pd.api.types.is_numeric_dtype(typed["qty"])
        assert typed["qty"].isna().sum() == 1


class TestPrimaryKeys:
    def test_detects_id_column(self, inferred) -> None:
        _, schema = inferred
        assert schema.primary_key is not None
        assert schema.primary_key.columns == ["order_id"]
        assert schema.primary_key.confidence >= 0.9

    def test_nulls_or_duplicates_disqualify(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 2, 3], "code": ["a", "b", None, "d"], "v": [1.5, 2.5, 3.5, 4.5]})
        schema = SchemaInferrer().infer(df)
        singles = [pk for pk in schema.primary_key_candidates if not pk.is_composite]
        assert singles == []

    def test_composite_key(self) -> None:
        df = pd.DataFrame({"store_id": [1, 1, 2, 2], "sku": ["a", "b", "a", "b"], "qty": [5, 5, 5, 6]})
        pk = SchemaInferrer().infer(df).primary_key
        assert pk is not None and pk.is_composite
        assert set(pk.columns) == {"store_id", "sku"}


class TestSchemaReport:
    def test_report_serializes(self, inferred) -> None:
        _, schema = inferred
        data = json.loads(json.dumps(schema.to_dict()))
        assert data["n_rows"] == 5
        assert data["type_summary"]["numeric"] == 2
        assert len(schema.to_dataframe()) == 6


class TestContract:
    def test_round_trip_from_report(self, inferred) -> None:
        _, schema = inferred
        contract = SchemaContract.from_schema_report(schema)
        restored = SchemaContract.from_json(contract.to_json())
        assert restored.columns["order_id"].unique is True
        assert restored.columns["order_date"].type == SemanticType.DATETIME

    @pytest.mark.parametrize("text, message", [
        ("not json", "not valid JSON"),
        ("[]", "JSON object"),
        ('{"columns": {}}', "non-empty"),
        ('{"columns": {"a": {"type": "decimal"}}}', "invalid type"),
        ('{"columns": {"a": {"nullable": "no"}}}', "true or false"),
        ('{"columns": {"a": {"colour": "red"}}}', "unknown keys"),
    ])
    def test_invalid_contracts(self, text: str, message: str) -> None:
        with pytest.raises(SchemaContractError, match=message):
            SchemaContract.from_json(text)
