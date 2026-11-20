"""End-to-end deterministic analysis pipeline.

``ingest → infer schema → cast → profile → validate quality``

This is the single service-layer entry point used by the Streamlit UI today and by
the FastAPI service and LangGraph agents in later milestones. Agents will *call* this
pipeline as a tool and reason over its structured output; they never re-implement or
override the deterministic computations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from datasentinel.ingestion import DatasetLoader, LoadedDataset
from datasentinel.profiling import DataProfiler, DatasetProfile
from datasentinel.quality import QualityConfig, QualityContext, QualityEngine, QualityReport
from datasentinel.schema import InferenceConfig, SchemaContract, SchemaInferrer, SchemaReport

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """All artefacts produced by one pipeline run."""

    dataset: LoadedDataset
    schema: SchemaReport
    typed_dataframe: pd.DataFrame
    profile: DatasetProfile
    quality: QualityReport

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        """The DataFrame exactly as ingested."""
        return self.dataset.dataframe

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary of every stage (excludes the DataFrames)."""
        return {
            "dataset": self.dataset.metadata.to_dict(),
            "schema": self.schema.to_dict(),
            "profile": self.profile.to_dict(),
            "quality": self.quality.to_dict(),
        }


def analyze_dataset(
    dataset: LoadedDataset,
    contract: Optional[SchemaContract] = None,
    quality_config: Optional[QualityConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> AnalysisResult:
    """Run schema inference, profiling and quality validation on a loaded dataset."""
    raw = dataset.dataframe
    inferrer = SchemaInferrer(inference_config)
    schema = inferrer.infer(raw)
    typed = inferrer.cast(raw, schema)
    profile = DataProfiler().profile(raw, typed, schema)
    context = QualityContext(raw=raw, typed=typed, schema=schema, profile=profile, contract=contract)
    quality = QualityEngine(quality_config).run(context, dataset_name=dataset.name)
    return AnalysisResult(dataset=dataset, schema=schema, typed_dataframe=typed, profile=profile, quality=quality)


def analyze_file(
    path: Union[str, Path],
    contract: Optional[SchemaContract] = None,
    quality_config: Optional[QualityConfig] = None,
    loader: Optional[DatasetLoader] = None,
) -> AnalysisResult:
    """Load a file from disk and analyse it."""
    dataset = (loader or DatasetLoader()).load_path(path)
    return analyze_dataset(dataset, contract=contract, quality_config=quality_config)
