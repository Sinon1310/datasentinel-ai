"""Validated loading of CSV and JSON datasets into pandas DataFrames.

Design notes
------------
* **Validate before parsing.** Extension and size are checked before any bytes are
  decoded, so oversized or unsupported uploads fail fast and cheaply.
* **Bytes in, DataFrame out.** :meth:`DatasetLoader.load_bytes` is the core entry point;
  it is equally usable from Streamlit uploads, FastAPI ``UploadFile`` objects or disk.
* **Graceful degradation.** Recoverable problems (malformed CSV rows, non-UTF-8
  encodings, nested JSON) produce *warnings* attached to the metadata instead of
  failing the whole load. Unrecoverable problems raise a specific
  :class:`~datasentinel.exceptions.IngestionError` subclass with a human-readable message.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, Optional, Tuple, Union

import pandas as pd

from datasentinel.config import get_settings
from datasentinel.exceptions import (
    DatasetParseError,
    EmptyDatasetError,
    FileSizeError,
    IngestionError,
    UnsupportedFileTypeError,
)
from datasentinel.ingestion.models import DatasetMetadata, LoadedDataset

logger = logging.getLogger(__name__)

_CSV_EXTENSIONS = frozenset({".csv"})
_JSON_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson"})
_SNIFF_DELIMITERS = ",;\t|"
_RECORD_CONTAINER_KEYS = ("data", "records", "rows", "items", "results")

Source = Union[str, Path, bytes, bytearray, BinaryIO]


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"  # pragma: no cover


class DatasetLoader:
    """Validate and load tabular files (CSV, JSON, JSON Lines).

    Args:
        max_size_bytes: Maximum accepted file size. Defaults to the configured setting.
        allowed_extensions: Accepted file extensions. Defaults to the configured setting.
    """

    def __init__(
        self,
        max_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[Iterable[str]] = None,
    ) -> None:
        settings = get_settings()
        self.max_size_bytes = int(max_size_bytes) if max_size_bytes is not None else settings.max_upload_size_bytes
        extensions = allowed_extensions if allowed_extensions is not None else settings.allowed_extensions
        self.allowed_extensions = tuple(sorted({e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}))

    # ------------------------------------------------------------------ public API

    def load(self, source: Source, filename: Optional[str] = None) -> LoadedDataset:
        """Load a dataset from a path, raw bytes, or a binary file-like object.

        Args:
            source: File path, bytes, or an object exposing ``read()`` (e.g. a Streamlit upload).
            filename: Required for raw bytes; otherwise inferred from the source.

        Raises:
            IngestionError: If the file is missing, unsupported, too large, empty or malformed.
        """
        if isinstance(source, (str, Path)):
            return self.load_path(source)
        if isinstance(source, (bytes, bytearray)):
            if not filename:
                raise IngestionError("A filename is required when loading raw bytes.")
            return self.load_bytes(bytes(source), filename)
        if hasattr(source, "read"):
            name = filename or getattr(source, "name", None)
            if not name:
                raise IngestionError("Could not determine a filename for the uploaded file.")
            declared_size = getattr(source, "size", None)
            if declared_size is not None:
                self._validate_extension(str(name))
                self._validate_size(int(declared_size), str(name))
            return self.load_bytes(source.read(), Path(str(name)).name)
        raise IngestionError(f"Unsupported source type: {type(source).__name__}")

    def load_path(self, path: Union[str, Path]) -> LoadedDataset:
        """Load a dataset from disk, validating extension and size before reading."""
        path = Path(path)
        if not path.is_file():
            raise IngestionError(f"File not found: {path}")
        self._validate_extension(path.name)
        self._validate_size(path.stat().st_size, path.name)
        return self.load_bytes(path.read_bytes(), path.name)

    def load_bytes(self, data: bytes, filename: str) -> LoadedDataset:
        """Validate and parse raw file bytes.

        Args:
            data: The full file contents.
            filename: Original filename; its extension selects the parser.
        """
        start = time.perf_counter()
        extension = self._validate_extension(filename)
        self._validate_size(len(data), filename)
        self._reject_binary(data, filename)

        text, encoding = self._decode(data)
        warnings: List[str] = []
        if encoding != "utf-8":
            warnings.append(f"File is not valid UTF-8; decoded using '{encoding}'.")
        if not text.strip():
            raise EmptyDatasetError(f"'{filename}' contains no data.")

        delimiter: Optional[str] = None
        json_layout: Optional[str] = None
        if extension in _CSV_EXTENSIONS:
            df, delimiter = self._parse_csv(text, filename, warnings)
            file_type = "csv"
        else:
            df, json_layout = self._parse_json(text, filename, warnings)
            file_type = "json"

        df = self._finalize(df, filename, warnings)
        elapsed = time.perf_counter() - start
        metadata = DatasetMetadata(
            filename=filename,
            file_type=file_type,
            size_bytes=len(data),
            encoding=encoding,
            n_rows=int(df.shape[0]),
            n_columns=int(df.shape[1]),
            load_seconds=round(elapsed, 4),
            delimiter=delimiter,
            json_layout=json_layout,
            warnings=warnings,
        )
        logger.info(
            "Loaded '%s' (%s, %s): %d rows x %d columns in %.3fs with %d warning(s)",
            filename, file_type, _human_size(len(data)), df.shape[0], df.shape[1], elapsed, len(warnings),
        )
        return LoadedDataset(dataframe=df, metadata=metadata)

    # ------------------------------------------------------------------ validation

    def _validate_extension(self, filename: str) -> str:
        extension = Path(filename).suffix.lower()
        if not extension:
            raise UnsupportedFileTypeError(
                f"'{filename}' has no file extension. Supported types: {', '.join(self.allowed_extensions)}."
            )
        if extension not in self.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension}'. Supported types: {', '.join(self.allowed_extensions)}."
            )
        if extension not in _CSV_EXTENSIONS | _JSON_EXTENSIONS:
            raise UnsupportedFileTypeError(f"No parser is available for '{extension}' files.")
        return extension

    def _validate_size(self, size: int, filename: str) -> None:
        if size == 0:
            raise EmptyDatasetError(f"'{filename}' is empty (0 bytes).")
        if size > self.max_size_bytes:
            raise FileSizeError(
                f"'{filename}' is {_human_size(size)}, which exceeds the maximum allowed size of "
                f"{_human_size(self.max_size_bytes)}."
            )

    @staticmethod
    def _reject_binary(data: bytes, filename: str) -> None:
        if b"\x00" in data[:8192]:
            raise UnsupportedFileTypeError(
                f"'{filename}' appears to be a binary file (or UTF-16 encoded), not a text CSV/JSON file."
            )

    @staticmethod
    def _decode(data: bytes) -> Tuple[str, str]:
        try:
            return data.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            pass
        try:
            return data.decode("cp1252"), "cp1252"
        except UnicodeDecodeError:
            return data.decode("latin-1"), "latin-1"

    # ------------------------------------------------------------------ CSV

    @staticmethod
    def _sniff_delimiter(text: str) -> str:
        sample = text[:65536]
        first_line = sample.splitlines()[0] if sample else ""
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=_SNIFF_DELIMITERS).delimiter
        except csv.Error:
            return ","
        return delimiter if delimiter in first_line else ","

    def _parse_csv(self, text: str, filename: str, warnings: List[str]) -> Tuple[pd.DataFrame, str]:
        delimiter = self._sniff_delimiter(text)
        if delimiter != ",":
            warnings.append(f"Detected '{repr(delimiter)[1:-1]}' as the CSV delimiter.")

        header = next(csv.reader(io.StringIO(text), delimiter=delimiter), [])
        stripped = [h.strip() for h in header]
        blank = [i + 1 for i, h in enumerate(stripped) if not h]
        if blank:
            warnings.append(f"Header has {len(blank)} blank column name(s) at position(s) {blank}.")
        duplicates = sorted({h for h in stripped if h and stripped.count(h) > 1})
        if duplicates:
            warnings.append(f"Header contains duplicate column names: {duplicates}. They were made unique.")

        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, low_memory=False)
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError(f"'{filename}' contains no parseable columns.") from exc
        except pd.errors.ParserError as exc:
            logger.warning("Strict CSV parse failed for '%s' (%s); retrying in tolerant mode.", filename, exc)
            df = self._parse_csv_tolerant(text, delimiter, filename, warnings, exc)
        return df, delimiter

    @staticmethod
    def _parse_csv_tolerant(
        text: str, delimiter: str, filename: str, warnings: List[str], original: Exception
    ) -> pd.DataFrame:
        bad_lines: List[List[str]] = []

        def _collect(line: List[str]) -> None:
            bad_lines.append(line)
            return None  # skip the row

        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python", on_bad_lines=_collect)
        except (pd.errors.ParserError, ValueError, csv.Error) as exc:
            raise DatasetParseError(f"'{filename}' is not a valid CSV file: {original}") from exc
        if bad_lines:
            example = delimiter.join(bad_lines[0])[:120]
            warnings.append(
                f"Skipped {len(bad_lines)} malformed row(s) with more fields than the header "
                f"(first example: '{example}')."
            )
        return df

    # ------------------------------------------------------------------ JSON

    def _parse_json(self, text: str, filename: str, warnings: List[str]) -> Tuple[pd.DataFrame, str]:
        stripped = text.strip()
        try:
            payload: Any = json.loads(stripped)
            layout = None
        except json.JSONDecodeError as exc:
            payload = self._parse_json_lines(stripped, filename, exc)
            layout = "json_lines"

        if isinstance(payload, list):
            df = self._frame_from_list(payload, filename)
            layout = layout or "records"
        elif isinstance(payload, dict):
            df, layout = self._frame_from_object(payload, filename, warnings)
        else:
            raise DatasetParseError(
                f"'{filename}' contains a JSON {type(payload).__name__}, but a list of records or an object is required."
            )
        return df, layout

    @staticmethod
    def _parse_json_lines(text: str, filename: str, original: json.JSONDecodeError) -> List[Any]:
        records: List[Any] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise DatasetParseError(
                    f"'{filename}' is not valid JSON: {original.msg} (line {original.lineno}, column {original.colno})."
                ) from original
        if len(records) < 2:
            raise DatasetParseError(
                f"'{filename}' is not valid JSON: {original.msg} (line {original.lineno}, column {original.colno})."
            )
        return records

    @staticmethod
    def _frame_from_list(records: List[Any], filename: str) -> pd.DataFrame:
        if not records:
            raise EmptyDatasetError(f"'{filename}' contains an empty JSON array.")
        if all(isinstance(r, dict) for r in records):
            return pd.json_normalize(records, sep=".")
        if all(isinstance(r, list) for r in records):
            return pd.DataFrame(records)
        if all(not isinstance(r, (dict, list)) for r in records):
            return pd.DataFrame({"value": records})
        raise DatasetParseError(
            f"'{filename}' mixes JSON objects, arrays and scalars at the top level; a consistent record list is required."
        )

    def _frame_from_object(self, obj: dict, filename: str, warnings: List[str]) -> Tuple[pd.DataFrame, str]:
        if not obj:
            raise EmptyDatasetError(f"'{filename}' contains an empty JSON object.")
        record_keys = [k for k, v in obj.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        preferred = [k for k in _RECORD_CONTAINER_KEYS if k in record_keys]
        if preferred or len(record_keys) == 1:
            key = preferred[0] if preferred else record_keys[0]
            warnings.append(f"Loaded records from the top-level JSON key '{key}'.")
            return pd.json_normalize(obj[key], sep="."), "nested_records"
        if all(isinstance(v, list) for v in obj.values()) and len({len(v) for v in obj.values()}) == 1:
            return pd.DataFrame(obj), "columnar"
        warnings.append("JSON object interpreted as a single record.")
        return pd.json_normalize([obj], sep="."), "single_record"

    # ------------------------------------------------------------------ normalisation

    def _finalize(self, df: pd.DataFrame, filename: str, warnings: List[str]) -> pd.DataFrame:
        if df.shape[1] == 0:
            raise EmptyDatasetError(f"'{filename}' has no columns.")
        if df.shape[0] == 0:
            raise EmptyDatasetError(f"'{filename}' has a header but no data rows.")

        df.columns = self._unique_column_names([str(c).strip() for c in df.columns])

        # Trailing delimiters create empty "Unnamed: N" columns; drop them.
        phantom = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
        if phantom:
            df = df.drop(columns=phantom)
            warnings.append(f"Dropped {len(phantom)} empty unnamed column(s) caused by trailing delimiters.")
            if df.shape[1] == 0:
                raise EmptyDatasetError(f"'{filename}' has no named columns with data.")

        # Nested lists / dicts are unhashable and break profiling; store them as JSON text.
        nested_columns = []
        for column in df.columns[df.dtypes == object]:
            if df[column].map(lambda v: isinstance(v, (list, dict))).any():
                df[column] = df[column].map(
                    lambda v: json.dumps(v, default=str, sort_keys=True) if isinstance(v, (list, dict)) else v
                )
                nested_columns.append(column)
        if nested_columns:
            warnings.append(f"Serialized nested list/object values to JSON text in columns: {nested_columns}.")
        return df.reset_index(drop=True)

    @staticmethod
    def _unique_column_names(names: List[str]) -> List[str]:
        seen: dict = {}
        result = []
        for index, name in enumerate(names):
            base = name or f"column_{index + 1}"
            candidate, suffix = base, 1
            while candidate in seen:
                suffix += 1
                candidate = f"{base}_{suffix}"
            seen[candidate] = True
            result.append(candidate)
        return result
