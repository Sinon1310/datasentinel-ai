"""DataSentinel AI — Pipeline Observability page (Streamlit ``pages/`` convention)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.dataset import render_dataset_summary  # noqa: E402
from app.components.shell import OBSERVABILITY, render_sidebar  # noqa: E402
from app.components.styles import inject_styles  # noqa: E402
from app.views import observability  # noqa: E402
from datasentinel.logging_config import configure_logging  # noqa: E402

configure_logging()
st.set_page_config(page_title="Pipeline Observability · DataSentinel AI", page_icon=":material/monitoring:",
                   layout="wide")
inject_styles()
render_sidebar(OBSERVABILITY, context=render_dataset_summary)
observability.render()
