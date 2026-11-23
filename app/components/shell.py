"""Application shell: brand, the single sidebar navigation, environment footer.

Navigation model: the five dataset sections live in the existing single entrypoint
(``app/streamlit_app.py``) and are selected with the ``?section=`` query parameter; Pipeline
Observability is a separate page (``app/pages/1_Pipeline_Observability.py``). Every item is an
``st.page_link`` rendered by :func:`render_sidebar`, so there is exactly one navigation system
(Streamlit's automatic page list is hidden in ``.streamlit/config.toml``).
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Callable

import streamlit as st

from datasentinel import __version__ as engine_version
from datasentinel.config import get_settings

MAIN_PAGE = "streamlit_app.py"
OBSERVABILITY_PAGE = "pages/1_Pipeline_Observability.py"
OBSERVABILITY = "observability"

# Widget values that must survive navigating to another page and back.
PERSISTENT_KEYS = ("sample_choice", "iqr", "null_tol")


@dataclass(frozen=True)
class NavItem:
    """One navigation destination."""

    key: str
    label: str
    icon: str
    page: str


NAV_ITEMS: tuple[NavItem, ...] = (
    NavItem("overview", "Overview", ":material/space_dashboard:", MAIN_PAGE),
    NavItem("quality", "Data Quality", ":material/fact_check:", MAIN_PAGE),
    NavItem("schema", "Schema Explorer", ":material/schema:", MAIN_PAGE),
    NavItem("profiling", "Data Profiling", ":material/query_stats:", MAIN_PAGE),
    NavItem("investigation", "AI Investigation", ":material/psychology:", MAIN_PAGE),
    NavItem(OBSERVABILITY, "Pipeline Observability", ":material/monitoring:", OBSERVABILITY_PAGE),
)
MAIN_SECTIONS = tuple(item.key for item in NAV_ITEMS if item.page == MAIN_PAGE)
NAV_GROUPS = {"overview": "Analyze", "quality": "Analyze", "schema": "Analyze", "profiling": "Analyze",
              "investigation": "Intelligence", OBSERVABILITY: "Operations"}


def current_section() -> str:
    """The active main-page section from the URL (defaults to Overview; unknown values fall back safely)."""
    value = st.query_params.get("section", "overview")
    return value if value in MAIN_SECTIONS else "overview"


def persist_widget_state() -> None:
    """Keep selected widget values when the user visits another page (Streamlit's documented pattern)."""
    for key in PERSISTENT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _nav_link(item: NavItem, active: bool) -> None:
    key = "nav_active" if active else f"nav_item_{item.key}"
    params = {"section": item.key} if item.page == MAIN_PAGE else None
    with st.container(key=key):
        try:
            st.page_link(item.page, label=item.label, icon=item.icon, query_params=params, width="stretch")
        except Exception:  # noqa: BLE001 — page registry unavailable (e.g. page rendered standalone in tests)
            st.markdown(f"**{item.label}**" if active else item.label)


def render_sidebar(active: str, context: Callable[[], None] | None = None) -> None:
    """Render the brand, navigation, optional page context and the environment footer."""
    persist_widget_state()
    with st.sidebar:
        st.markdown(
            '<div class="ds-brand"><div class="ds-brand-mark"><svg width="18" height="18" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            'aria-hidden="true"><path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z"/>'
            '<path d="M8 12.5l2.5 2.5L16 9.5"/></svg></div><div>'
            '<div class="ds-brand-name">DataSentinel<span class="ds-brand-ai">AI</span></div>'
            '<div class="ds-brand-tag">Data engineering intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        group = None
        for item in NAV_ITEMS:
            if NAV_GROUPS[item.key] != group:
                group = NAV_GROUPS[item.key]
                st.markdown(f'<div class="ds-nav-label">{group}</div>', unsafe_allow_html=True)
            _nav_link(item, active == item.key)
        st.markdown('<div class="ds-side-sep"></div>', unsafe_allow_html=True)
        if context is not None:
            context()
        _footer()


def _footer() -> None:
    settings = get_settings()
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    # Reports configuration only; connectivity is never implied (no API call is made to check it).
    ai_dot, ai_state = ("green", f"API key set · {settings.groq_model}") if key.strip() else ("amber", "API key not set")
    st.markdown(
        '<div class="ds-side-sep"></div><div class="ds-env">'
        '<span class="ds-dot ds-dot-slate"></span><b>Environment</b> Local workspace<br>'
        f'<span class="ds-dot ds-dot-{ai_dot}"></span><b>AI</b> {html.escape(ai_state)}<br>'
        f'<span class="ds-dot ds-dot-slate"></span><b>Engine</b> v{engine_version}</div>',
        unsafe_allow_html=True,
    )
