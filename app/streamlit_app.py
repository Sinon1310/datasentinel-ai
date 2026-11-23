"""DataSentinel AI — Streamlit entry point.

Run with::

    streamlit run app/streamlit_app.py

This script hosts the dataset-centred sections (Overview, Data Quality, Schema Explorer, Data
Profiling, AI Investigation), selected with the ``?section=`` query parameter from the single
sidebar navigation. Pipeline Observability lives in ``app/pages/1_Pipeline_Observability.py``.
The UI is a thin presentation layer: all computation happens in the engine
(:func:`datasentinel.pipeline.analyze_dataset`, cached per dataset and settings).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.dataset import current_baseline, current_dataset, load_analysis, quality_settings, \
    render_dataset_controls  # noqa: E402
from app.components.investigation import get_service, render_investigation  # noqa: E402
from app.components.shell import current_section, render_sidebar  # noqa: E402
from app.components.styles import inject_styles  # noqa: E402
from app.views import overview, profiling, quality, schema  # noqa: E402
from datasentinel.logging_config import configure_logging  # noqa: E402

logger = configure_logging()


def main() -> None:
    """Render the active section."""
    st.set_page_config(page_title="DataSentinel AI", page_icon=":material/shield:", layout="wide")
    inject_styles()
    section = current_section()
    render_sidebar(section, context=render_dataset_controls)

    if section == "investigation":
        dataset = current_dataset()
        iqr, null_tol = quality_settings()
        render_investigation(dataset.as_file if dataset else None, current_baseline(), iqr, null_tol)
        return

    state = load_analysis()
    if section == "quality":
        quality.render(state)
    elif section == "schema":
        schema.render(state)
    elif section == "profiling":
        profiling.render(state)
    else:
        overview.render(state, has_baseline=current_baseline() is not None, ai_configured=get_service() is not None)


main()
