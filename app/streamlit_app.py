"""DataSentinel AI — Streamlit dashboard (Milestone 1: Data Intelligence Engine).

Run with::

    streamlit run app/streamlit_app.py

The UI is a thin presentation layer: all computation happens in
:func:`datasentinel.pipeline.analyze_dataset`, which is cached per (file, settings).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import sections  # noqa: E402
from app.components.styles import inject_styles  # noqa: E402
from datasentinel import __version__  # noqa: E402
from datasentinel.config import get_settings  # noqa: E402
from datasentinel.exceptions import DataSentinelError, IngestionError, SchemaContractError  # noqa: E402
from datasentinel.ingestion import DatasetLoader  # noqa: E402
from datasentinel.logging_config import configure_logging  # noqa: E402
from datasentinel.pipeline import AnalysisResult, analyze_dataset  # noqa: E402
from datasentinel.quality import QualityConfig  # noqa: E402
from datasentinel.schema import SchemaContract  # noqa: E402

SAMPLES = {
    "Customer orders (CSV)": ROOT / "sample_data" / "customer_orders.csv",
    "Sensor readings (nested JSON)": ROOT / "sample_data" / "sensor_readings.json",
}

logger = configure_logging()


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(
    data: bytes,
    filename: str,
    contract_json: Optional[str],
    iqr_multiplier: float,
    max_null_pct: float,
) -> AnalysisResult:
    """Ingest and analyse a dataset. Cached on file content and settings."""
    contract = SchemaContract.from_json(contract_json) if contract_json else None
    config = QualityConfig(iqr_multiplier=iqr_multiplier, max_null_pct=max_null_pct)
    dataset = DatasetLoader().load_bytes(data, filename)
    return analyze_dataset(dataset, contract=contract, quality_config=config)


def sidebar() -> Tuple[Optional[Tuple[bytes, str]], Optional[str], float, float]:
    """Render the sidebar and return (file, contract_json, iqr_multiplier, max_null_pct)."""
    settings = get_settings()
    with st.sidebar:
        st.markdown("### 📥 Dataset")
        upload = st.file_uploader(
            "Upload CSV or JSON",
            type=[e.lstrip(".") for e in settings.allowed_extensions],
            help=f"Maximum size: {settings.max_upload_size_mb:.0f} MB",
        )
        sample = st.selectbox("…or use a sample dataset", ["—", *SAMPLES], key="sample_choice")

        st.markdown("### 📐 Schema contract")
        contract_file = st.file_uploader("Expected schema (JSON, optional)", type=["json"], key="contract",
                                         help="Download one from the Schema Explorer tab, edit it, and upload it here.")

        st.markdown("### ⚙️ Quality settings")
        iqr = st.slider("IQR multiplier (outlier fence)", 1.0, 3.0, 1.5, 0.1)
        null_tol = st.slider("Missing-value tolerance (%)", 0.0, 20.0, 0.0, 0.5,
                             help="Columns with a missing share at or below this pass the completeness check.")
        st.divider()
        st.caption(f"DataSentinel AI v{__version__} · Milestone 1")

    file: Optional[Tuple[bytes, str]] = None
    if upload is not None:
        file = (upload.getvalue(), upload.name)
    elif sample != "—":
        path = SAMPLES[sample]
        if path.exists():
            file = (path.read_bytes(), path.name)
        else:
            st.sidebar.error(f"Sample file missing: {path.name}. Run scripts/generate_sample_data.py.")

    contract_json = contract_file.getvalue().decode("utf-8", errors="replace") if contract_file else None
    return file, contract_json, iqr, null_tol


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="DataSentinel AI", page_icon="🛡️", layout="wide")
    inject_styles()
    st.markdown('<div class="ds-hero"><h1>🛡️ DataSentinel AI</h1><span class="ds-badge">Data Intelligence Engine</span>'
                '</div><div class="ds-subtitle">Schema inference · profiling · deterministic data quality validation'
                '</div>', unsafe_allow_html=True)

    file, contract_json, iqr, null_tol = sidebar()
    if file is None:
        sections.render_welcome()
        return

    data, filename = file
    try:
        with st.spinner(f"Analysing {filename}…"):
            result = run_analysis(data, filename, contract_json, iqr, null_tol)
    except IngestionError as exc:
        st.error(f"**Could not load the dataset.** {exc}")
        return
    except SchemaContractError as exc:
        st.error(f"**Invalid schema contract.** {exc}")
        return
    except DataSentinelError as exc:
        st.error(f"**Analysis failed.** {exc}")
        return
    except Exception:  # noqa: BLE001 — last-resort guard so the UI never shows a raw traceback
        logger.exception("Unexpected error while analysing '%s'", filename)
        st.error("An unexpected error occurred while analysing the dataset. Check the server logs for details.")
        return

    if contract_json:
        st.caption("📐 Validating against the uploaded schema contract.")
    sections.render_kpis(result)
    tabs = st.tabs([
        "Overview",
        "Schema Explorer",
        f"Data Quality ({sections.failed_count(result)} issues)",
        "Profiling",
        "Data Preview",
    ])
    with tabs[0]:
        sections.render_overview(result)
    with tabs[1]:
        sections.render_schema(result)
    with tabs[2]:
        sections.render_quality(result)
    with tabs[3]:
        sections.render_profiling(result)
    with tabs[4]:
        sections.render_preview(result)


main()
