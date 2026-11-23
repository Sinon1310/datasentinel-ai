"""Design tokens and the small amount of global CSS the theme cannot express.

Colours live in ``.streamlit/config.toml`` (native theming, including the dark sidebar).
This module holds the same tokens for Plotly charts and inline HTML, plus minimal CSS for
layout polish. Selectors target stable Streamlit ``data-testid`` attributes or keyed
containers (``st-key-*``) only.
"""

from __future__ import annotations

import streamlit as st

# Workspace
INK = "#0F172A"
INK_MUTED = "#64748B"
SURFACE = "#FFFFFF"
CANVAS = "#F8FAFC"
BORDER = "#E2E8F0"
GRID = "rgba(100,116,139,0.15)"

# Navigation
NAV_BG = "#111827"
NAV_SURFACE = "#1F2937"
NAV_BORDER = "#374151"
NAV_TEXT = "#F9FAFB"
NAV_MUTED = "#9CA3AF"

# Accents
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
VIOLET = "#7C3AED"
CYAN = "#0891B2"

# Diverging scale (correlations): red ↔ neutral ↔ blue
DIVERGING_NEG = "#DC2626"
DIVERGING_MID = "#F1F5F9"
DIVERGING_POS = "#2563EB"

# Reserved status palette — always paired with a text label or icon, never colour alone.
STATUS_COLORS = {
    "passed": "#16A34A",
    "success": "#16A34A",
    "low": "#D97706",
    "medium": "#EA580C",
    "high": "#DC2626",
    "critical": "#991B1B",
    "failed": "#DC2626",
    "skipped": "#94A3B8",
    "cancelled": "#94A3B8",
    "pending": "#0891B2",
    "running": "#2563EB",
    "stale": "#D97706",
}
# st.badge colours for the same semantics.
BADGE_COLORS = {
    "passed": "green", "success": "green", "completed": "green", "confirmed": "green", "done": "green",
    "failed": "red", "critical": "red", "high": "red", "blocked": "red", "rejected": "red",
    "medium": "orange", "low": "orange", "warning": "orange", "stale": "orange", "degraded": "orange",
    "hypothesis": "orange", "awaiting_approval": "orange", "waiting": "orange", "incomplete": "orange",
    "interrupted": "orange",
    "running": "blue", "active": "blue", "pending": "blue", "info": "blue", "approved": "blue",
    "skipped": "gray", "cancelled": "gray", "not_started": "gray", "not_reached": "gray",
}
SEVERITY_ORDER = ["critical", "high", "medium", "low"]

_CSS = """
<style>
  /* ---------- Workspace rhythm & type ---------- */
  /* Wide but bounded: uses large desktops without stretching tables/cards to unreadable widths. */
  [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1640px;
                                         padding-left: 2.25rem; padding-right: 2.25rem; }
  @media (max-width: 900px) { [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; } }
  [data-testid="stMain"] h1 { letter-spacing: -0.02em; line-height: 1.15; }
  [data-testid="stMain"] h2, [data-testid="stMain"] h3 { letter-spacing: -0.01em; }
  [data-testid="stMain"] [data-testid="stCaptionContainer"] { color: #64748B; }

  /* ---------- Page header ---------- */
  .ds-page-eyebrow { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
                     margin-bottom: 0.1rem; }
  .ds-accent-blue { color: #2563EB; }
  .ds-accent-violet { color: #6D28D9; }
  .ds-page-subtitle { color: #64748B; font-size: 0.95rem; margin: -0.4rem 0 0.55rem 0; max-width: 92ch; }
  .ds-meta-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 0.35rem 0; }
  .ds-meta { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.78rem; color: #334155;
             background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.15rem 0.5rem; }
  .ds-meta-k { color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.66rem; letter-spacing: 0.05em; }

  /* ---------- KPI cards ---------- */
  [data-testid="stMetric"] { background: #FFFFFF; padding: 0.8rem 1rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.7rem !important; color: #64748B; font-weight: 700;
                                    text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
  [class*="st-key-kpis"] { display: grid !important; gap: 0.6rem; grid-template-columns: repeat(4, minmax(0, 1fr)); }
  [class*="st-key-kpis"] > * { width: auto !important; min-width: 0; }
  [class*="st-key-kpis6_"] { grid-template-columns: repeat(6, minmax(0, 1fr)); }
  [class*="st-key-kpis5_"] { grid-template-columns: repeat(5, minmax(0, 1fr)); }
  [class*="st-key-kpis3_"] { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  [class*="st-key-kpis7_"] { grid-template-columns: repeat(7, minmax(0, 1fr)); }
  @media (max-width: 1700px) { [class*="st-key-kpis7_"] { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
  @media (max-width: 1520px) { [class*="st-key-kpis6_"] { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 1300px) { [class*="st-key-kpis5_"] { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 640px) { [class*="st-key-kpis"] { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; } }
  [class*="st-key-kpic_"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
                            padding: 0.8rem 1rem 0.75rem 1rem; height: 100%; justify-content: flex-start; }
  [class*="st-key-kpic_"] [data-testid="stMetric"] { padding: 0; background: transparent; }
  .ds-kpi-caption { font-size: 0.76rem; color: #64748B; line-height: 1.35; margin-top: 0.1rem; }

  /* ---------- Cards ---------- */
  [class*="st-key-card_"] { background: #FFFFFF; }

  /* ---------- Sidebar ---------- */
  [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.25rem; }
  .ds-brand { display: flex; align-items: center; gap: 0.65rem; margin: 0 0 0.35rem 0.1rem; }
  .ds-brand-mark { width: 32px; height: 32px; border-radius: 8px; background: #2563EB; color: #FFFFFF;
                   display: flex; align-items: center; justify-content: center;
                   box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12); }
  .ds-brand-name { font-weight: 700; font-size: 1.02rem; color: #F9FAFB; line-height: 1.1; letter-spacing: -0.01em; }
  .ds-brand-ai { font-size: 0.62rem; font-weight: 700; color: #DDD6FE; background: rgba(124,58,237,0.28);
                 border-radius: 4px; padding: 0.05rem 0.3rem; margin-left: 0.25rem; vertical-align: 2px; }
  .ds-brand-tag { font-size: 0.72rem; color: #9CA3AF; margin-top: 0.1rem; }
  .ds-nav-label { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                  color: #9CA3AF; margin: 0.85rem 0 0.15rem 0.55rem; }
  [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { border-radius: 6px; padding-top: 0.28rem;
                                                                 padding-bottom: 0.28rem; }
  [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p { color: #CBD5E1; font-weight: 500; font-size: 0.9rem; }
  [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: #94A3B8; }
  /* Neutralise Streamlit's per-page highlight; highlight only the active section */
  [data-testid="stSidebar"] [class*="st-key-nav_item_"] [data-testid="stPageLink-NavLink"] { background: transparent; }
  [data-testid="stSidebar"] [class*="st-key-nav_item_"] [data-testid="stPageLink-NavLink"]:hover { background: #1F2937; }
  [data-testid="stSidebar"] [class*="st-key-nav_item_"] [data-testid="stPageLink-NavLink"]:hover p { color: #F9FAFB; }
  [data-testid="stSidebar"] .st-key-nav_active [data-testid="stPageLink-NavLink"] {
      background: #1F2937; box-shadow: inset 3px 0 0 #3B82F6; }
  [data-testid="stSidebar"] .st-key-nav_active [data-testid="stPageLink-NavLink"] p { color: #FFFFFF; font-weight: 600; }
  [data-testid="stSidebar"] .st-key-nav_active [data-testid="stIconMaterial"] { color: #60A5FA; }
  .ds-side-sep { border-top: 1px solid #1F2937; margin: 0.9rem 0 0.2rem 0; }
  .ds-ds-card { background: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 0.55rem 0.7rem;
                margin: 0.1rem 0 0.4rem 0; }
  .ds-ds-name { color: #F9FAFB; font-weight: 600; font-size: 0.85rem; word-break: break-all; }
  .ds-ds-meta { color: #9CA3AF; font-size: 0.72rem; margin-top: 0.1rem; }
  .ds-ds-empty { color: #9CA3AF; font-size: 0.8rem; margin: 0.1rem 0 0.4rem 0.1rem; }
  .ds-env { font-size: 0.72rem; color: #9CA3AF; line-height: 1.75; }
  .ds-env b { color: #D1D5DB; font-weight: 600; }
  .ds-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.4rem; vertical-align: 1px; }
  .ds-dot-green { background: #22C55E; } .ds-dot-amber { background: #F59E0B; } .ds-dot-slate { background: #64748B; }

  /* ---------- Empty states ---------- */
  .ds-empty { display: flex; gap: 0.9rem; align-items: flex-start; border: 1px solid #E2E8F0; border-radius: 10px;
              background: #FFFFFF; padding: 1rem 1.15rem; margin-bottom: 0.75rem; }
  .ds-empty-icon { flex: 0 0 auto; width: 40px; height: 40px; border-radius: 9px; display: flex; align-items: center;
                   justify-content: center; }
  .ds-tone-blue .ds-empty-icon { background: #EFF6FF; color: #2563EB; }
  .ds-tone-violet .ds-empty-icon { background: #F5F3FF; color: #6D28D9; }
  .ds-tone-amber .ds-empty-icon { background: #FFFBEB; color: #B45309; }
  .ds-tone-slate .ds-empty-icon { background: #F1F5F9; color: #475569; }
  .ds-empty h3 { margin: 0.1rem 0 0.2rem 0; padding: 0; font-size: 1.02rem; color: #0F172A; }
  .ds-empty p { color: #475569; margin: 0 0 0.35rem 0; font-size: 0.9rem; }
  .ds-empty p.ds-empty-why { color: #64748B; font-size: 0.85rem; }
  .ds-steps { margin: 0.2rem 0 0 0; padding-left: 1.1rem; color: #334155; font-size: 0.88rem; }
  .ds-steps li { margin: 0.12rem 0; }

  /* ---------- Demo / AI accents ---------- */
  .ds-demo-banner { display: flex; gap: 0.55rem; align-items: center; border: 1px solid #DDD6FE; background: #F5F3FF;
                    color: #4C1D95; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.86rem; margin-bottom: 0.75rem; }
  .ds-demo-banner svg { flex: 0 0 auto; color: #6D28D9; }

  /* ---------- Investigation workflow timeline ---------- */
  .ds-flow { display: flex; flex-wrap: wrap; gap: 0.35rem 0.25rem; align-items: stretch; margin: 0.15rem 0 0.25rem 0; }
  .ds-step { display: flex; align-items: center; gap: 0.4rem; border: 1px solid #E2E8F0; background: #FFFFFF;
             border-radius: 7px; padding: 0.3rem 0.55rem 0.3rem 0.35rem; font-size: 0.8rem; color: #334155; }
  .ds-step-n { width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
               font-size: 0.7rem; font-weight: 700; flex: 0 0 auto; }
  .ds-step-s { color: #64748B; font-size: 0.72rem; }
  .ds-step.done .ds-step-n { background: #DCFCE7; color: #166534; }
  .ds-step.failed .ds-step-n { background: #FEE2E2; color: #991B1B; }
  .ds-step.failed { border-color: #FCA5A5; }
  .ds-step.degraded .ds-step-n, .ds-step.interrupted .ds-step-n { background: #FEF3C7; color: #92400E; }
  .ds-step.waiting { border-color: #F59E0B; background: #FFFBEB; }
  .ds-step.waiting .ds-step-n { background: #F59E0B; color: #FFFFFF; }
  .ds-step.skipped, .ds-step.not_reached { color: #64748B; background: #F8FAFC; }
  .ds-step.skipped .ds-step-n, .ds-step.not_reached .ds-step-n { background: #F1F5F9; color: #94A3B8; }
  .ds-step-arrow { color: #CBD5E1; align-self: center; font-size: 0.8rem; }

  /* Findings / evidence / approval accents (keyed containers) */
  [class*="st-key-finding_confirmed_"] { background: #FFFFFF; box-shadow: inset 3px 0 0 #16A34A; }
  [class*="st-key-finding_hypothesis_"] { background: #FFFFFF; box-shadow: inset 3px 0 0 #D97706; }
  [class*="st-key-approval_panel"] { background: #FFFBEB; border-color: #F59E0B !important; }
</style>
"""


def inject_styles() -> None:
    """Inject global CSS once per render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge_color(state: str) -> str:
    """The ``st.badge`` colour for a status/severity keyword."""
    return BADGE_COLORS.get(str(state).lower(), "gray")
