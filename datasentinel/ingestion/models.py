"""Data models produced by the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from datasentinel.utils.serialization import to_builtin


@dataclass
class DatasetMetadata:
    """Provenance and load diagnostics for an ingested dataset."""

    filename: str
    file_type: str
    size_bytes: int
    encoding: str
    n_rows: int
    n_columns: int
    load_seconds: float
    delimiter: Optional[str] = None
    json_layout: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_builtin(self)


@dataclass
class LoadedDataset:
    """A successfully ingested dataset: the raw DataFrame plus its metadata."""

    dataframe: pd.DataFrame
    metadata: DatasetMetadata

    @property
    def name(self) -> str:
        """The source filename."""
        return self.metadata.filename
