"""Dataset session context: sidebar controls, cross-page persistence and cached analysis.

The active dataset is kept in ``st.session_state`` (bytes + name) so it survives navigation to
the Pipeline Observability page, where the sidebar upload widget is not rendered. Analysis is
computed by the engine and cached per (content, settings), so changing sections never re-profiles.
"""

from __future__ import annotations

import hashlib
import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from datasentinel.config import get_settings
from datasentinel.exceptions import DataSentinelError, IngestionError, SchemaContractError
from datasentinel.ingestion import DatasetLoader
from datasentinel.pipeline import AnalysisResult, analyze_dataset
from datasentinel.quality import QualityConfig
from datasentinel.schema.drift import load_baseline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {
    "Customer orders (CSV)": ROOT / "sample_data" / "customer_orders.csv",
    "Sensor readings (nested JSON)": ROOT / "sample_data" / "sensor_readings.json",
}
NO_SAMPLE = "—"
_DATASET = "ds_dataset"
_BASELINE = "ds_baseline"


@dataclass(frozen=True)
class ActiveDataset:
    """The dataset currently loaded in this browser session."""

    data: bytes
    filename: str
    source: str  # "upload" | "sample"
    loaded_at: str
    digest: str

    @property
    def as_file(self) -> tuple[bytes, str]:
        """``(bytes, filename)`` as expected by the investigation service."""
        return self.data, self.filename


@dataclass(frozen=True)
class AnalysisState:
    """Outcome of analysing the active dataset."""

    dataset: ActiveDataset | None
    result: AnalysisResult | None
    error_title: str | None = None
    error_detail: str | None = None


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(data: bytes, filename: str, contract_json: str | None, iqr_multiplier: float,
                 max_null_pct: float) -> AnalysisResult:
    """Ingest and analyse a dataset. Cached on file content and settings (no cross-dataset reuse)."""
    contract = load_baseline(contract_json)[0] if contract_json else None
    config = QualityConfig(iqr_multiplier=iqr_multiplier, max_null_pct=max_null_pct)
    dataset = DatasetLoader().load_bytes(data, filename)
    return analyze_dataset(dataset, contract=contract, quality_config=config)


def _set_dataset(data: bytes, filename: str, source: str) -> None:
    st.session_state[_DATASET] = ActiveDataset(
        data=data, filename=filename, source=source,
        loaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        digest=hashlib.sha256(data).hexdigest()[:12],
    )


def _on_upload() -> None:
    upload = st.session_state.get("upload")
    if upload is not None:
        _set_dataset(upload.getvalue(), upload.name, "upload")


def _on_sample() -> None:
    choice = st.session_state.get("sample_choice", NO_SAMPLE)
    if choice != NO_SAMPLE and choice in SAMPLES and SAMPLES[choice].exists():
        _set_dataset(SAMPLES[choice].read_bytes(), SAMPLES[choice].name, "sample")


def _on_baseline() -> None:
    upload = st.session_state.get("contract")
    st.session_state[_BASELINE] = upload.getvalue().decode("utf-8", errors="replace") if upload is not None else None


def _clear() -> None:
    st.session_state.pop(_DATASET, None)
    st.session_state.pop(_BASELINE, None)
    st.session_state["sample_choice"] = NO_SAMPLE


def current_dataset() -> ActiveDataset | None:
    """The active dataset, if any."""
    return st.session_state.get(_DATASET)


def current_baseline() -> str | None:
    """The active schema baseline JSON text, if any."""
    return st.session_state.get(_BASELINE)


def quality_settings() -> tuple[float, float]:
    """``(iqr_multiplier, max_null_pct)`` from the sidebar (defaults when not yet rendered)."""
    return float(st.session_state.get("iqr", 1.5)), float(st.session_state.get("null_tol", 0.0))


def render_dataset_controls() -> None:
    """Sidebar dataset controls (rendered on the main page sections)."""
    settings = get_settings()
    st.markdown('<div class="ds-nav-label">Dataset</div>', unsafe_allow_html=True)
    st.file_uploader("Upload CSV or JSON", type=[e.lstrip(".") for e in settings.allowed_extensions], key="upload",
                     on_change=_on_upload, help=f"Maximum size: {settings.max_upload_size_mb:.0f} MB")
    st.selectbox("Or load a sample dataset", [NO_SAMPLE, *SAMPLES], key="sample_choice", on_change=_on_sample)
    dataset = current_dataset()
    if dataset is not None:
        _dataset_card(dataset)
        st.button("Clear dataset", key="clear_dataset", on_click=_clear, width="stretch")
    with st.expander("Schema baseline & quality settings"):
        st.file_uploader("Schema contract or exported schema (JSON)", type=["json"], key="contract",
                         on_change=_on_baseline,
                         help="Validates the dataset against a contract and enables drift detection.")
        # Defaults live in session state (not as widget arguments) so values persist across pages without conflicts.
        st.session_state.setdefault("iqr", 1.5)
        st.session_state.setdefault("null_tol", 0.0)
        st.slider("IQR multiplier (outlier fence)", min_value=1.0, max_value=3.0, step=0.1, key="iqr")
        st.slider("Missing-value tolerance (%)", min_value=0.0, max_value=20.0, step=0.5, key="null_tol",
                  help="Columns with a missing share at or below this pass the completeness check.")


def render_dataset_summary() -> None:
    """Read-only dataset context (used on pages without dataset controls)."""
    dataset = current_dataset()
    st.markdown('<div class="ds-nav-label">Dataset</div>', unsafe_allow_html=True)
    if dataset:
        _dataset_card(dataset)
    else:
        st.markdown('<div class="ds-ds-empty">No dataset loaded</div>', unsafe_allow_html=True)


def _dataset_card(dataset: ActiveDataset) -> None:
    """Compact sidebar summary of the active dataset (file name is escaped; it is user-controlled)."""
    size = len(dataset.data)
    size_text = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
    st.markdown(f'<div class="ds-ds-card"><div class="ds-ds-name">{html.escape(dataset.filename)}</div>'
                f'<div class="ds-ds-meta">{html.escape(dataset.source)} · {size_text} · loaded {dataset.loaded_at}</div>'
                '</div>', unsafe_allow_html=True)


def load_analysis() -> AnalysisState:
    """Analyse the active dataset (cached), translating expected errors into explanations."""
    dataset = current_dataset()
    if dataset is None:
        return AnalysisState(None, None)
    iqr, null_tol = quality_settings()
    try:
        with st.spinner(f"Analysing {dataset.filename}…"):
            result = run_analysis(dataset.data, dataset.filename, current_baseline(), iqr, null_tol)
    except IngestionError as exc:
        return AnalysisState(dataset, None, "Could not load the dataset.", str(exc))
    except SchemaContractError as exc:
        return AnalysisState(dataset, None, "Invalid schema baseline.", str(exc))
    except DataSentinelError as exc:
        return AnalysisState(dataset, None, "Analysis failed.", str(exc))
    except Exception:  # noqa: BLE001 — last-resort guard so the UI never shows a raw traceback
        logger.exception("Unexpected error while analysing '%s'", dataset.filename)
        return AnalysisState(dataset, None, "Unexpected error.", "Check the server logs for details.")
    return AnalysisState(dataset, result)
