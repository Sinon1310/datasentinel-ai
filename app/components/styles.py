"""Visual tokens and global CSS for the dashboard."""

from __future__ import annotations

import streamlit as st

# Validated reference palette (dataviz skill): sequential blue + fixed status colors.
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
DIVERGING_NEG = "#e34948"
DIVERGING_MID = "#f0efec"
DIVERGING_POS = "#2a78d6"
STATUS_COLORS = {
    "passed": "#0ca30c",
    "low": "#fab219",
    "medium": "#ec835a",
    "high": "#d03b3b",
    "critical": "#9b1c1c",
    "skipped": "#9a9893",
}
STATUS_ICONS = {
    "passed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "info": "ℹ️",
    "low": "🟡",
    "medium": "🟠",
    "high": "🔴",
    "critical": "⛔",
}
SEVERITY_ORDER = ["critical", "high", "medium", "low"]

_CSS = """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  .ds-hero { display:flex; align-items:baseline; gap:.75rem; margin-bottom:.25rem; }
  .ds-hero h1 { margin:0; padding:0; font-size:1.9rem; }
  .ds-subtitle { opacity:.7; margin-bottom:1.25rem; font-size:.95rem; }
  .ds-badge { display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.75rem;
              font-weight:600; border:1px solid rgba(128,128,128,.35); opacity:.85; }
  .ds-score { font-size:3rem; font-weight:700; line-height:1; }
  .ds-grade { font-size:1rem; font-weight:600; opacity:.75; margin-left:.35rem; }
  .ds-muted { opacity:.65; font-size:.85rem; }
  div[data-testid="stMetric"] { padding: .75rem 1rem; }
  div[data-testid="stTabs"] button p { font-size: .95rem; font-weight: 600; }
</style>
"""


def inject_styles() -> None:
    """Inject global CSS once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)
