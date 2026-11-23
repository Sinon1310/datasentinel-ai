"""Plotly figure builders. Pure functions: prepared data in, figure out — no Streamlit calls,
no engine computation. Status colours are reserved and always paired with text labels."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from app.components.styles import (
    BLUE,
    DIVERGING_MID,
    DIVERGING_NEG,
    DIVERGING_POS,
    GRID,
    INK,
    INK_MUTED,
    SEVERITY_ORDER,
    STATUS_COLORS,
    SURFACE,
)
from datasentinel.profiling.models import CategoricalStats, DatasetProfile
from datasentinel.quality.models import CheckStatus, QualityReport
from datasentinel.schema.models import SchemaReport

_FONT = dict(family="Source Sans Pro, Inter, system-ui, sans-serif", size=12, color=INK)


def _base_layout(fig: go.Figure, title: str | None, height: int = 300, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK)) if title else None,
        margin=dict(l=8, r=20, t=44 if title else 12, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=_FONT,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        bargap=0.35,
        barcornerradius=4,
        hoverlabel=dict(font_size=12, bgcolor=SURFACE, bordercolor="#CBD5E1", font_color=INK),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#CBD5E1",
                     tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_MUTED, size=12))
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=INK_MUTED),
                     title_font=dict(color=INK_MUTED, size=12))
    return fig


def empty_figure(message: str, title: str | None = None, height: int = 220) -> go.Figure:
    """A labelled empty chart (no fabricated data)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13, color=INK_MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, title, height=height)


def _hbar(labels: Sequence[str], values: Sequence[float], title: str, x_title: str, colors=None,
          text: Sequence[str] | None = None, hover: str = "<b>%{y}</b><br>%{x}<extra></extra>",
          customdata=None, x_range=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation="h", marker_color=colors or BLUE,
        text=list(text) if text is not None else list(values), textposition="outside", cliponaxis=False,
        customdata=customdata, hovertemplate=hover,
    ))
    if x_range is None and values:
        # Headroom so outside value labels are never clipped at the right edge.
        x_range = [0, max(values) * 1.2 if max(values) > 0 else 1]
    fig.update_xaxes(title=x_title, range=x_range)
    fig.update_yaxes(type="category", automargin=True)
    return _base_layout(fig, title, height=max(220, 70 + 30 * len(labels)))


# ---------------------------------------------------------------------- data quality


def missing_values_chart(profile: DatasetProfile) -> go.Figure:
    """Missing-value share for columns that have any."""
    rows = sorted(((p.name, p.null_pct, p.null_count) for p in profile.columns.values() if p.null_count),
                  key=lambda t: t[1])
    if not rows:
        return empty_figure("No missing values in any column", "Missing values by column")
    names, pcts, counts = zip(*rows)
    fig = _hbar(names, pcts, "Missing values by column", "Missing (% of rows)", text=[f"{p:.1f}%" for p in pcts],
                customdata=counts, hover="<b>%{y}</b><br>%{x:.2f}% missing<br>%{customdata} rows<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    return fig


def type_distribution_chart(schema: SchemaReport) -> go.Figure:
    """Columns per inferred semantic type."""
    summary = sorted(schema.type_summary().items(), key=lambda kv: kv[1])
    fig = _hbar([k for k, _ in summary], [v for _, v in summary], "Columns by inferred type", "Columns",
                hover="<b>%{y}</b><br>%{x} column(s)<extra></extra>")
    fig.update_xaxes(dtick=1)
    return fig


def dimension_scores_chart(report: QualityReport) -> go.Figure:
    """Score per quality dimension on a fixed 0–100 axis (unevaluated dimensions omitted)."""
    rows = sorted((d for d in report.dimension_scores if d.score is not None), key=lambda d: d.score)
    if not rows:
        return empty_figure("No dimensions were evaluated", "Score by quality dimension")
    return _hbar(
        [d.dimension.value.capitalize() for d in rows], [d.score for d in rows], "Score by quality dimension",
        "Score (0–100)", text=[f"{d.score:.1f}" for d in rows], x_range=[0, 108],
        customdata=[[d.effective_weight * 100, d.checks_evaluated, d.checks_failed] for d in rows],
        hover="<b>%{y}</b><br>Score %{x:.1f}<br>Weight %{customdata[0]:.0f}%<br>"
              "%{customdata[1]} checks, %{customdata[2]} failed<extra></extra>",
    )


def severity_chart(report: QualityReport) -> go.Figure:
    """Failed checks by severity (reserved status colours + labels)."""
    counts = report.summary()["failed_by_severity"]
    labels = [s for s in reversed(SEVERITY_ORDER) if counts.get(s)]
    if not labels:
        return empty_figure("No failed checks", "Failed checks by severity")
    fig = _hbar([s.capitalize() for s in labels], [counts[s] for s in labels], "Failed checks by severity",
                "Failed checks", colors=[STATUS_COLORS[s] for s in labels],
                hover="<b>%{y}</b><br>%{x} failed check(s)<extra></extra>")
    fig.update_xaxes(dtick=1)
    return fig


def check_status_chart(report: QualityReport) -> go.Figure:
    """Checks by outcome: passed / failed / skipped."""
    by_status = report.summary()["by_status"]
    order = [CheckStatus.SKIPPED.value, CheckStatus.FAILED.value, CheckStatus.PASSED.value]
    labels = [s for s in order if by_status.get(s)]
    if not labels:
        return empty_figure("No checks were run", "Checks by outcome")
    fig = _hbar([s.capitalize() for s in labels], [by_status[s] for s in labels], "Checks by outcome", "Checks",
                colors=[STATUS_COLORS[s] for s in labels], hover="<b>%{y}</b><br>%{x} check(s)<extra></extra>")
    fig.update_xaxes(dtick=max(1, max(by_status.values()) // 8))
    return fig


def duplicates_chart(n_rows: int, duplicate_rows: int) -> go.Figure:
    """Unique vs duplicate rows (duplicates = exact copies of an earlier row)."""
    if n_rows == 0:
        return empty_figure("No rows", "Row uniqueness")
    labels, values = ["Duplicate rows", "Unique rows"], [duplicate_rows, n_rows - duplicate_rows]
    return _hbar(labels, values, "Row uniqueness", "Rows", colors=[STATUS_COLORS["failed"], BLUE],
                 text=[f"{v:,} ({v / n_rows * 100:.1f}%)" for v in values],
                 hover="<b>%{y}</b><br>%{x:,} rows<extra></extra>")


# ---------------------------------------------------------------------- profiling


def histogram_chart(values: pd.Series, name: str, lower: Optional[float] = None,
                    upper: Optional[float] = None) -> go.Figure:
    """Histogram of a numeric column with optional IQR fences (from the quality engine)."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return empty_figure("No numeric values", f"Distribution of {name}")
    fig = go.Figure(go.Histogram(x=values, nbinsx=40, marker_color=BLUE, marker_line=dict(width=1, color=SURFACE),
                                 hovertemplate="%{x}<br>%{y} rows<extra></extra>"))
    for bound, label in ((lower, "lower fence"), (upper, "upper fence")):
        if bound is not None and values.min() <= bound <= values.max():
            fig.add_vline(x=bound, line_dash="dash", line_width=1.5, line_color=INK_MUTED,
                          annotation_text=label, annotation_position="top", annotation_font_color=INK_MUTED)
    fig.update_xaxes(showgrid=False, title=name)
    fig.update_yaxes(title="Rows", showgrid=True, gridcolor=GRID)
    fig.update_layout(bargap=0.04)
    return _base_layout(fig, f"Distribution of {name}", height=320)


def box_chart(values: pd.Series, name: str) -> go.Figure:
    """Horizontal box plot (Tukey whiskers)."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    fig = go.Figure(go.Box(x=values, name=name, marker_color=BLUE, boxpoints="outliers", marker_size=7,
                           line_width=2, hoverinfo="x"))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(title=name)
    return _base_layout(fig, "Box plot (points beyond whiskers are IQR outliers)", height=190)


def top_values_chart(stats: CategoricalStats, name: str) -> go.Figure:
    """Most frequent values of a categorical/text column."""
    items = list(reversed(stats.top_values))
    return _hbar([str(i.value)[:40] for i in items], [i.count for i in items], f"Top values of {name}", "Rows",
                 text=[f"{i.pct:.1f}%" for i in items], customdata=[i.pct for i in items],
                 hover="<b>%{y}</b><br>%{x:,} rows (%{customdata:.1f}%)<extra></extra>")


def datetime_trend_chart(values: pd.Series, name: str) -> go.Figure:
    """Row counts over time, bucketed by a resolution suited to the span."""
    values = pd.to_datetime(values, errors="coerce").dropna()
    if values.empty:
        return empty_figure("No valid dates", f"{name} over time")
    span_days = (values.max() - values.min()).days
    freq, label = ("D", "day") if span_days <= 92 else (("W", "week") if span_days <= 730 else ("M", "month"))
    counts = values.dt.to_period(freq).dt.start_time.value_counts().sort_index()
    fig = go.Figure(go.Scatter(
        x=counts.index, y=counts.values, mode="lines+markers", line=dict(color=BLUE, width=2),
        marker=dict(size=7, color=BLUE, line=dict(width=1.5, color=SURFACE)),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y}} rows per {label}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title=f"Rows per {label}", showgrid=True, gridcolor=GRID, rangemode="tozero")
    fig.update_xaxes(showgrid=False, title=name)
    return _base_layout(fig, f"{name} over time", height=300)


def correlation_heatmap(typed: pd.DataFrame, numeric_columns: list) -> go.Figure:
    """Pearson correlations on a diverging red ↔ blue scale with a neutral midpoint."""
    if len(numeric_columns) < 2:
        return empty_figure("At least two numeric columns are needed", "Correlation matrix (Pearson)")
    corr = typed[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, DIVERGING_NEG], [0.5, DIVERGING_MID], [1, DIVERGING_POS]], xgap=2, ygap=2,
        text=corr.round(2).values, texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=10, title="r")))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, "Correlation matrix (Pearson)", height=max(300, 80 + 42 * len(numeric_columns)))


# ---------------------------------------------------------------------- pipeline observability

RUN_STATUS_ORDER = ["SUCCESS", "FAILED", "CANCELLED", "RUNNING", "PENDING"]


def status_by_pipeline_chart(frame: pd.DataFrame) -> go.Figure:
    """Stacked horizontal bars: executions per pipeline by status. ``frame``: pipeline, status, count."""
    if frame.empty:
        return empty_figure("No executions in the selected range", "Executions by pipeline and status")
    fig = go.Figure()
    for status in RUN_STATUS_ORDER:
        part = frame[frame["status"] == status]
        if not part.empty:
            fig.add_trace(go.Bar(y=part["pipeline"], x=part["count"], name=status.capitalize(), orientation="h",
                                 marker_color=STATUS_COLORS[status.lower()],
                                 marker_line=dict(width=1, color=SURFACE),
                                 hovertemplate=f"<b>%{{y}}</b><br>{status.capitalize()}: %{{x}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Executions")
    fig.update_yaxes(type="category", automargin=True)
    return _base_layout(fig, "Executions by pipeline and status",
                        height=max(240, 90 + 34 * frame["pipeline"].nunique()), legend=True)


def run_volume_chart(frame: pd.DataFrame) -> go.Figure:
    """Stacked daily executions by outcome. ``frame``: bucket, success, failed, cancelled, active."""
    if frame.empty:
        return empty_figure("No executions in the selected range", "Run volume over time")
    fig = go.Figure()
    for column, label, color in (("success", "Success", STATUS_COLORS["success"]),
                                 ("failed", "Failed", STATUS_COLORS["failed"]),
                                 ("cancelled", "Cancelled", STATUS_COLORS["cancelled"]),
                                 ("active", "Running / pending", STATUS_COLORS["running"])):
        fig.add_trace(go.Bar(x=frame["bucket"], y=frame[column], name=label, marker_color=color,
                             marker_line=dict(width=1, color=SURFACE),
                             hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y}}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.2, hovermode="x unified")
    fig.update_yaxes(title="Executions", showgrid=True, gridcolor=GRID)
    fig.update_xaxes(showgrid=False, title="Day (UTC)")
    return _base_layout(fig, "Run volume over time", height=300, legend=True)


def line_trend_chart(frame: pd.DataFrame, columns: Sequence[tuple[str, str, str]], title: str, y_title: str,
                     y_suffix: str = "") -> go.Figure:
    """Line trend(s) over ``frame['bucket']``. ``columns``: (column, label, colour)."""
    if frame.empty or all(frame[c].isna().all() for c, _, _ in columns):
        return empty_figure("No data in the selected range", title)
    fig = go.Figure()
    for column, label, color in columns:
        fig.add_trace(go.Scatter(x=frame["bucket"], y=frame[column], name=label, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=6, color=color),
                                 connectgaps=False,
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:,.1f}}{y_suffix}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor=GRID, rangemode="tozero")
    fig.update_xaxes(showgrid=False, title="Day (UTC)")
    return _base_layout(fig, title, height=300, legend=len(columns) > 1)


def failure_category_chart(categories: Sequence[str], counts: Sequence[int]) -> go.Figure:
    """Failures by classified category (magnitude: single hue)."""
    if not categories:
        return empty_figure("No failures in the selected range", "Failures by category")
    pairs = sorted(zip(categories, counts), key=lambda t: t[1])
    fig = _hbar([c.replace("_", " ").title() for c, _ in pairs], [n for _, n in pairs], "Failures by category",
                "Failed executions", hover="<b>%{y}</b><br>%{x} failure(s)<extra></extra>")
    fig.update_xaxes(dtick=max(1, max(counts) // 8))
    return fig


def task_timeline_chart(frame: pd.DataFrame) -> go.Figure:
    """Gantt-style task timeline. ``frame``: label, start, end, status."""
    if frame.empty:
        return empty_figure("No task executions recorded", "Task timeline")
    fig = go.Figure()
    for status in frame["status"].unique():
        part = frame[frame["status"] == status]
        durations = (part["end"] - part["start"]).dt.total_seconds() * 1000
        fig.add_trace(go.Bar(
            y=part["label"], x=durations, base=part["start"], orientation="h", name=status.capitalize(),
            marker_color=STATUS_COLORS.get(status.lower(), BLUE), marker_line=dict(width=1, color=SURFACE),
            customdata=part[["start", "end"]].astype(str).values,
            hovertemplate="<b>%{y}</b><br>" + status.capitalize() +
                          "<br>%{customdata[0]} → %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(type="date", title="Time (UTC)")
    fig.update_yaxes(type="category", autorange="reversed", automargin=True)
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Task timeline", height=max(200, 80 + 34 * len(frame)), legend=True)
