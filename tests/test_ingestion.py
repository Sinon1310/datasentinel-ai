"""Tests for dataset ingestion and file validation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from datasentinel.exceptions import (
    DatasetParseError,
    EmptyDatasetError,
    FileSizeError,
    IngestionError,
    UnsupportedFileTypeError,
)
from datasentinel.ingestion import DatasetLoader


class TestValidCsv:
    def test_loads_csv_from_path(self, loader: DatasetLoader, clean_csv_path: Path) -> None:
        dataset = loader.load_path(clean_csv_path)
        assert dataset.dataframe.shape == (5, 6)
        assert list(dataset.dataframe.columns) == ["order_id", "customer", "amount", "order_date", "is_active", "category"]
        assert dataset.metadata.file_type == "csv"
        assert dataset.metadata.delimiter == ","
        assert dataset.metadata.encoding == "utf-8"
        assert dataset.metadata.warnings == []

    def test_loads_csv_from_file_like_object(self, loader: DatasetLoader, clean_csv_path: Path) -> None:
        buffer = io.BytesIO(clean_csv_path.read_bytes())
        buffer.name = "upload.csv"
        dataset = loader.load(buffer)
        assert dataset.metadata.filename == "upload.csv"
        assert dataset.metadata.n_rows == 5

    def test_detects_semicolon_delimiter(self, loader: DatasetLoader) -> None:
        dataset = loader.load_bytes(b"a;b;c\n1;2;3\n4;5;6\n", "semi.csv")
        assert list(dataset.dataframe.columns) == ["a", "b", "c"]
        assert dataset.metadata.delimiter == ";"

    def test_handles_utf8_bom_and_latin1(self, loader: DatasetLoader) -> None:
        bom = loader.load_bytes("﻿name,city\nJosé,São Paulo\n".encode("utf-8"), "bom.csv")
        assert list(bom.dataframe.columns) == ["name", "city"]
        latin = loader.load_bytes("name,city\nJosé,Zürich\n".encode("latin-1"), "latin.csv")
        assert latin.dataframe.loc[0, "city"] == "Zürich"
        assert latin.metadata.encoding != "utf-8"
        assert any("UTF-8" in w for w in latin.metadata.warnings)

    def test_skips_malformed_rows_with_warning(self, loader: DatasetLoader) -> None:
        data = b"a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n"
        dataset = loader.load_bytes(data, "ragged.csv")
        assert len(dataset.dataframe) == 2
        assert any("malformed" in w for w in dataset.metadata.warnings)

    def test_drops_phantom_unnamed_columns(self, loader: DatasetLoader) -> None:
        dataset = loader.load_bytes(b"a,b,\n1,2,\n3,4,\n", "trailing.csv")
        assert list(dataset.dataframe.columns) == ["a", "b"]


class TestValidJson:
    def test_loads_record_list(self, loader: DatasetLoader) -> None:
        payload = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]).encode()
        dataset = loader.load_bytes(payload, "records.json")
        assert dataset.dataframe.shape == (2, 2)
        assert dataset.metadata.json_layout == "records"

    def test_flattens_nested_objects_and_serializes_lists(self, loader: DatasetLoader) -> None:
        payload = json.dumps({"data": [{"id": 1, "meta": {"x": 1}, "tags": ["a", "b"]},
                                       {"id": 2, "meta": {"x": 2}, "tags": []}]}).encode()
        dataset = loader.load_bytes(payload, "nested.json")
        assert "meta.x" in dataset.dataframe.columns
        assert dataset.dataframe.loc[0, "tags"] == '["a", "b"]'
        assert dataset.metadata.json_layout == "nested_records"

    def test_loads_json_lines(self, loader: DatasetLoader) -> None:
        payload = b'{"id": 1}\n{"id": 2}\n{"id": 3}\n'
        dataset = loader.load_bytes(payload, "events.jsonl")
        assert len(dataset.dataframe) == 3
        assert dataset.metadata.json_layout == "json_lines"

    def test_loads_columnar_object(self, loader: DatasetLoader) -> None:
        dataset = loader.load_bytes(b'{"a": [1, 2], "b": ["x", "y"]}', "cols.json")
        assert dataset.dataframe.shape == (2, 2)


class TestInvalidFiles:
    def test_rejects_unsupported_extension(self, loader: DatasetLoader) -> None:
        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
            loader.load_bytes(b"a,b\n1,2\n", "data.xlsx")

    def test_rejects_missing_extension(self, loader: DatasetLoader) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            loader.load_bytes(b"a,b\n1,2\n", "data")

    def test_rejects_oversized_file(self) -> None:
        small_loader = DatasetLoader(max_size_bytes=10)
        with pytest.raises(FileSizeError, match="exceeds"):
            small_loader.load_bytes(b"a,b\n1,2\n3,4\n", "big.csv")

    def test_oversized_path_rejected_before_reading(self, tmp_path: Path) -> None:
        path = tmp_path / "big.csv"
        path.write_text("a,b\n" + "1,2\n" * 100)
        with pytest.raises(FileSizeError):
            DatasetLoader(max_size_bytes=50).load_path(path)

    def test_rejects_empty_file(self, loader: DatasetLoader) -> None:
        with pytest.raises(EmptyDatasetError):
            loader.load_bytes(b"", "empty.csv")

    def test_rejects_whitespace_only_file(self, loader: DatasetLoader) -> None:
        with pytest.raises(EmptyDatasetError):
            loader.load_bytes(b"   \n\n  ", "blank.csv")

    def test_rejects_header_only_csv(self, loader: DatasetLoader) -> None:
        with pytest.raises(EmptyDatasetError, match="no data rows"):
            loader.load_bytes(b"a,b,c\n", "header.csv")

    def test_rejects_binary_content(self, loader: DatasetLoader) -> None:
        with pytest.raises(UnsupportedFileTypeError, match="binary"):
            loader.load_bytes(b"PK\x03\x04\x00\x00binary", "fake.csv")

    def test_rejects_malformed_json(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetParseError, match="not valid JSON"):
            loader.load_bytes(b'[{"id": 1}, {"id": 2', "broken.json")

    def test_rejects_json_scalar(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetParseError):
            loader.load_bytes(b"42", "scalar.json")

    def test_rejects_empty_json_array(self, loader: DatasetLoader) -> None:
        with pytest.raises(EmptyDatasetError):
            loader.load_bytes(b"[]", "empty.json")

    def test_missing_path(self, loader: DatasetLoader, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not found"):
            loader.load_path(tmp_path / "nope.csv")

    def test_bytes_without_filename(self, loader: DatasetLoader) -> None:
        with pytest.raises(IngestionError, match="filename"):
            loader.load(b"a,b\n1,2\n")

    def test_all_errors_share_base_class(self, loader: DatasetLoader) -> None:
        for data, name in ((b"", "x.csv"), (b"a", "x.txt"), (b"{", "x.json")):
            with pytest.raises(IngestionError):
                loader.load_bytes(data, name)
