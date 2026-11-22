"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest

from datasentinel.ingestion import DatasetLoader, LoadedDataset
from datasentinel.pipeline import AnalysisResult, analyze_dataset
from datasentinel.profiling import DataProfiler
from datasentinel.quality import QualityConfig, QualityContext
from datasentinel.schema import SchemaContract, SchemaInferrer

CLEAN_CSV = """order_id,customer,amount,order_date,is_active,category
1,alice,10.5,2024-01-01,yes,A
2,bob,20.0,2024-01-02,no,B
3,carol,15.25,2024-01-03,yes,A
4,dave,12.0,2024-01-04,no,C
5,erin,18.75,2024-01-05,yes,B
"""


LIVE_CALLS_DISABLED = "Live Groq API calls are disabled in the offline test suite (use -m live)."


@pytest.fixture(autouse=True)
def _block_live_llm_calls(request, monkeypatch):
    """Guarantee the offline suite never reaches the Groq API or consumes tokens, even if a key is configured."""
    if request.node.get_closest_marker("live"):
        return
    try:
        from groq.resources.chat import completions
    except ImportError:  # agent extras not installed
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError(LIVE_CALLS_DISABLED)

    monkeypatch.setattr(completions.Completions, "create", _blocked)
    monkeypatch.setattr(completions.AsyncCompletions, "create", _blocked)


@pytest.fixture
def loader() -> DatasetLoader:
    """A loader with a 1 MB size limit."""
    return DatasetLoader(max_size_bytes=1024 * 1024)


@pytest.fixture
def clean_csv_path(tmp_path: Path) -> Path:
    """A small, clean CSV on disk."""
    path = tmp_path / "clean.csv"
    path.write_text(CLEAN_CSV)
    return path


@pytest.fixture
def make_context() -> Callable[..., QualityContext]:
    """Build a QualityContext from a DataFrame, running real inference and profiling."""

    def _make(df: pd.DataFrame, contract: Optional[SchemaContract] = None) -> QualityContext:
        inferrer = SchemaInferrer()
        schema = inferrer.infer(df)
        typed = inferrer.cast(df, schema)
        profile = DataProfiler().profile(df, typed, schema)
        return QualityContext(raw=df, typed=typed, schema=schema, profile=profile, contract=contract)

    return _make


@pytest.fixture
def analyze() -> Callable[..., AnalysisResult]:
    """Run the full pipeline on an in-memory DataFrame."""

    def _analyze(df: pd.DataFrame, config: Optional[QualityConfig] = None,
                 contract: Optional[SchemaContract] = None) -> AnalysisResult:
        csv_bytes = df.to_csv(index=False).encode()
        dataset: LoadedDataset = DatasetLoader().load_bytes(csv_bytes, "frame.csv")
        return analyze_dataset(dataset, contract=contract, quality_config=config)

    return _analyze


@pytest.fixture
def normal_values() -> np.ndarray:
    """200 well-behaved values with no extreme tail."""
    rng = np.random.default_rng(0)
    return np.round(rng.uniform(40, 60, 200), 2)
