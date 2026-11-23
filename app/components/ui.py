"""Reusable presentation components. No engine computation happens here."""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import streamlit as st

from app.components.styles import badge_color

# Minimal stroke icons (inline SVG; Streamlit's icon font is not available to raw HTML).
_ICON_PATHS = {
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
                '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "cloud_off": '<path d="M3 3l18 18"/><path d="M17.5 17.5H7a4.5 4.5 0 0 1-1.3-8.8"/>'
                 '<path d="M9.2 5.4A6 6 0 0 1 18 9.5a4 4 0 0 1 2.4 7"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
    "sparkles": '<path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7-4.7-1.8 4.7-1.8z"/>'
                '<path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="M10.8 12.2L20 3"/><path d="M17 6l3 3"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    "flask": '<path d="M9 3h6"/><path d="M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3"/><path d="M7.5 15h9"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
}

# Status words with a glyph so tables never rely on colour alone.
_STATUS_GLYPHS = {
    "passed": "✓", "success": "✓", "succeeded": "✓", "completed": "✓", "done": "✓", "confirmed": "✓",
    "failed": "✕", "error": "✕", "blocked": "✕", "rejected": "✕",
    "skipped": "○", "not_reached": "○",
    "running": "●", "active": "●",
    "pending": "◷", "waiting": "◷", "awaiting_approval": "◷",
    "stale": "⚠", "degraded": "⚠", "interrupted": "⚠", "hypothesis": "?",
    "cancelled": "■",
}


@dataclass(frozen=True)
class Kpi:
    """One KPI card. ``value`` is pre-formatted; ``caption`` is a one-line context; ``help`` explains the formula."""

    label: str
    value: str
    help: str | None = None
    delta: str | float | None = None
    delta_color: str = "normal"
    caption: str | None = None


def icon_svg(name: str, size: int = 20) -> str:
    """Inline SVG for one of the bundled stroke icons."""
    paths = _ICON_PATHS.get(name, _ICON_PATHS["database"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>')


def status_text(status: str) -> str:
    """Status label for tables: glyph + capitalised word (e.g. ``✕ Failed``)."""
    word = str(status).replace("_", " ").strip()
    glyph = _STATUS_GLYPHS.get(str(status).lower().split(" ")[0], "")
    return f"{glyph} {word[:1].upper()}{word[1:].lower()}".strip()


def _meta_chip(item: str) -> str:
    label, _, value = item.partition(" · ")
    if not value:
        return f'<span class="ds-meta">{html.escape(label)}</span>'
    return f'<span class="ds-meta"><span class="ds-meta-k">{html.escape(label)}</span>{html.escape(value)}</span>'


def page_header(title: str, subtitle: str, eyebrow: str | None = None, context: Sequence[str] = (),
                accent: str = "blue", actions: Callable[[], None] | None = None) -> None:
    """Consistent page header: eyebrow, title, subtitle, metadata chips and optional right-aligned actions."""
    def _text() -> None:
        if eyebrow:
            st.markdown(f'<div class="ds-page-eyebrow ds-accent-{html.escape(accent)}">{html.escape(eyebrow)}</div>',
                        unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.markdown(f'<div class="ds-page-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
        if context:
            st.markdown('<div class="ds-meta-row">' + "".join(_meta_chip(c) for c in context) + "</div>",
                        unsafe_allow_html=True)

    if actions is None:
        _text()
        return
    left, right = st.columns([5, 2], vertical_alignment="bottom")
    with left:
        _text()
    with right, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        actions()


def kpi_row(kpis: Iterable[Kpi]) -> None:
    """KPI cards in a count-aware responsive grid. Values use native ``st.metric`` (accessible, testable)."""
    items = list(kpis)
    if not items:
        return
    digest = hashlib.sha1("|".join(k.label for k in items).encode()).hexdigest()[:10]
    # The card count is part of the key so CSS can choose balanced grids (6 → 6×1 / 3×2 / 2×3, 5 → 5 / 3+2 / 2s).
    with st.container(horizontal=True, gap="small", key=f"kpis{len(items)}_{digest}"):
        for index, kpi in enumerate(items):
            with st.container(key=f"kpic_{digest}_{index}", gap=None):
                st.metric(kpi.label, kpi.value, delta=kpi.delta, delta_color=kpi.delta_color, help=kpi.help,
                          width="stretch")
                if kpi.caption:
                    st.markdown(f'<div class="ds-kpi-caption">{html.escape(kpi.caption)}</div>',
                                unsafe_allow_html=True)


def card(name: str, parent=None):
    """A bordered white card. ``name`` must be unique within the page (it becomes a stable CSS hook)."""
    return (parent or st).container(border=True, key=f"card_{name}")


def fmt_number(value: float | int | None) -> str:
    """Readable statistic: separators for large values, no scientific notation in normal ranges."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1e15 or (0 < magnitude < 1e-4):
        return f"{value:.3e}"
    if float(value).is_integer():
        return f"{value:,.0f}"
    if magnitude >= 1000:
        return f"{value:,.1f}"
    return f"{value:,.4g}" if magnitude >= 1 else f"{value:.4f}".rstrip("0").rstrip(".")


def display_value(value) -> str:
    """Table cell text: numbers with separators, missing values as an em dash (never the word 'None')."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def section(title: str, caption: str | None = None) -> None:
    """Section heading with an optional one-line explanation."""
    st.subheader(title, anchor=False)
    if caption:
        st.caption(caption)


def status_badge(label: str, state: str | None = None, icon: str | None = None) -> None:
    """A status badge; colour follows the reserved status palette, the label always carries the meaning."""
    st.badge(label, color=badge_color(state or label), icon=icon)  # type: ignore[arg-type]


def badge_markdown(label: str, state: str | None = None) -> str:
    """Inline badge markdown (for use inside text, tables-as-markdown, expander bodies)."""
    safe = str(label).replace("[", "(").replace("]", ")")
    return f":{badge_color(state or label)}-badge[{safe}]"


def empty_state(title: str, body: str, steps: Sequence[str] = (), icon: str = "database", tone: str = "blue",
                why: str | None = None) -> None:
    """Onboarding-style empty state: what is missing, why it matters, what to do next."""
    items = "".join(f"<li>{html.escape(s)}</li>" for s in steps)
    steps_html = f'<ol class="ds-steps">{items}</ol>' if items else ""
    why_html = f'<p class="ds-empty-why">{html.escape(why)}</p>' if why else ""
    st.markdown(
        f'<div class="ds-empty ds-tone-{html.escape(tone)}"><div class="ds-empty-icon">{icon_svg(icon, 22)}</div>'
        f'<div class="ds-empty-body"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p>{why_html}'
        f'{steps_html}</div></div>',
        unsafe_allow_html=True)


def error_state(title: str, detail: str) -> None:
    """An error that explains what happened (never a raw traceback)."""
    st.error(f"**{title}** {detail}")


def demo_banner(text: str) -> None:
    """Persistent label for synthetic / demo content."""
    st.markdown(f'<div class="ds-demo-banner">{icon_svg("flask", 16)}<span><strong>Synthetic demo data.</strong> '
                f'{html.escape(text)}</span></div>', unsafe_allow_html=True)


def fmt_int(value: int | float | None) -> str:
    """Thousands-separated integer, or an em dash when unavailable (never a fabricated 0)."""
    return "—" if value is None else f"{int(value):,}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Percentage string, or an em dash when unavailable."""
    return "—" if value is None else f"{value:.{digits}f}%"


def fmt_score(value: float | None) -> str:
    """Quality score; ``None`` (no checks evaluated) stays unavailable, never 0."""
    return "—" if value is None else f"{value:.1f}"


def fmt_duration_ms(value: int | float | None) -> str:
    """Human-readable duration from milliseconds."""
    if value is None:
        return "—"
    seconds = value / 1000
    if seconds < 1:
        return f"{value:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_bytes(num: float) -> str:
    """File-size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"
