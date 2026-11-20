"""Plotly figure builders. Pure functions: data in, figure out — no Streamlit calls."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from app.components.styles import (
    BLUE,
    DIVERGING_MID,
    DIVERGING_NEG,
    DIVERGING_POS,
    SEVERITY_ORDER,
    STATUS_COLORS,
)
from datasentinel.profiling.models import CategoricalStats, DatasetProfile
from datasentinel.quality.models import CheckStatus, QualityReport
from datasentinel.schema.models import SchemaReport


def _base_layout(fig: go.Figure, height: int = 320, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        margin=dict(l=8, r=16, t=40 if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.35,
        barcornerradius=4,
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _empty(message: str, height: int = 220) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13), opacity=0.6)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=height)


def missing_values_chart(profile: DatasetProfile) -> go.Figure:
    """Horizontal bars of missing-value percentage for columns that have any."""
    data = sorted(((p.name, p.null_pct, p.null_count) for p in profile.columns.values() if p.null_count),
                  key=lambda t: t[1])
    if not data:
        return _empty("No missing values in any column 🎉")
    names, pcts, counts = zip(*data)
    fig = go.Figure(go.Bar(
        x=pcts, y=names, orientation="h", marker_color=BLUE, customdata=counts,
        text=[f"{p:.1f}%" for p in pcts], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% missing<br>%{customdata} rows<extra></extra>",
    ))
    fig.update_xaxes(ticksuffix="%", rangemode="tozero")
    return _base_layout(fig, height=max(220, 40 + 28 * len(names)), title="Missing values by column")


def type_distribution_chart(schema: SchemaReport) -> go.Figure:
    """Count of columns per semantic type."""
    summary = sorted(schema.type_summary().items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in summary], y=[k for k, _ in summary], orientation="h", marker_color=BLUE,
        text=[v for _, v in summary], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} column(s)<extra></extra>",
    ))
    fig.update_xaxes(dtick=1, rangemode="tozero")
    return _base_layout(fig, height=max(200, 40 + 34 * len(summary)), title="Columns by semantic type")


def dimension_scores_chart(report: QualityReport) -> go.Figure:
    """Score per quality dimension on a fixed 0–100 axis."""
    rows = [d for d in report.dimension_scores if d.score is not None]
    if not rows:
        return _empty("No dimensions evaluated")
    rows = sorted(rows, key=lambda d: d.score)
    fig = go.Figure(go.Bar(
        x=[d.score for d in rows], y=[d.dimension.value.capitalize() for d in rows], orientation="h",
        marker_color=BLUE, text=[f"{d.score:.1f}" for d in rows], textposition="outside", cliponaxis=False,
        customdata=[[d.effective_weight * 100, d.checks_evaluated, d.checks_failed] for d in rows],
        hovertemplate=("<b>%{y}</b><br>Score %{x:.1f}<br>Weight %{customdata[0]:.0f}%"
                       "<br>%{customdata[1]} checks, %{customdata[2]} failed<extra></extra>"),
    ))
    fig.update_xaxes(range=[0, 108], dtick=20)
    return _base_layout(fig, height=260, title="Score by quality dimension")


def severity_chart(report: QualityReport) -> go.Figure:
    """Failed checks by severity, colored with the reserved status palette and labelled."""
    counts = report.summary()["failed_by_severity"]
    labels = [s for s in SEVERITY_ORDER if counts.get(s)]
    if not labels:
        return _empty("No failed checks ✅")
    labels = list(reversed(labels))
    fig = go.Figure(go.Bar(
        x=[counts[s] for s in labels], y=[s.capitalize() for s in labels], orientation="h",
        marker_color=[STATUS_COLORS[s] for s in labels], text=[counts[s] for s in labels],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} failed check(s)<extra></extra>",
    ))
    fig.update_xaxes(dtick=1, rangemode="tozero")
    return _base_layout(fig, height=max(200, 40 + 36 * len(labels)), title="Failed checks by severity")


def histogram_chart(values: pd.Series, name: str, lower: Optional[float] = None,
                    upper: Optional[float] = None) -> go.Figure:
    """Histogram of a numeric column with optional IQR fences."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return _empty("No numeric values")
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=40, marker_color=BLUE, marker_line=dict(width=1, color="rgba(255,255,255,0.9)"),
        hovertemplate="%{x}<br>%{y} rows<extra></extra>",
    ))
    for bound, label in ((lower, "lower fence"), (upper, "upper fence")):
        if bound is not None and values.min() <= bound <= values.max():
            fig.add_vline(x=bound, line_dash="dash", line_width=1.5, line_color="rgba(128,128,128,0.8)",
                          annotation_text=label, annotation_position="top")
    fig.update_yaxes(title="rows", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(showgrid=False, title=name)
    fig.update_layout(bargap=0.04)
    return _base_layout(fig, height=320, title=f"Distribution of {name}")


def box_chart(values: pd.Series, name: str) -> go.Figure:
    """Horizontal box plot (Tukey whiskers) for a numeric column."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    fig = go.Figure(go.Box(
        x=values, name=name, marker_color=BLUE, boxpoints="outliers", marker_size=8,
        line_width=2, hoverinfo="x",
    ))
    fig.update_yaxes(showticklabels=False)
    return _base_layout(fig, height=180, title="Box plot (points beyond whiskers are IQR outliers)")


def top_values_chart(stats: CategoricalStats, name: str) -> go.Figure:
    """Most frequent values of a categorical column."""
    items = list(reversed(stats.top_values))
    fig = go.Figure(go.Bar(
        x=[i.count for i in items], y=[str(i.value)[:40] for i in items], orientation="h", marker_color=BLUE,
        customdata=[i.pct for i in items], text=[f"{i.pct:.1f}%" for i in items], textposition="outside",
        cliponaxis=False, hovertemplate="<b>%{y}</b><br>%{x} rows (%{customdata:.1f}%)<extra></extra>",
    ))
    fig.update_yaxes(type="category")
    return _base_layout(fig, height=max(220, 40 + 28 * len(items)), title=f"Top values of {name}")


def datetime_trend_chart(values: pd.Series, name: str) -> go.Figure:
    """Row counts over time, bucketed by a resolution suited to the date span."""
    values = pd.to_datetime(values, errors="coerce").dropna()
    if values.empty:
        return _empty("No valid dates")
    span_days = (values.max() - values.min()).days
    freq, label = ("D", "day") if span_days <= 92 else (("W", "week") if span_days <= 730 else ("M", "month"))
    counts = values.dt.to_period(freq).dt.start_time.value_counts().sort_index()
    fig = go.Figure(go.Scatter(
        x=counts.index, y=counts.values, mode="lines+markers", line=dict(color=BLUE, width=2),
        marker=dict(size=8, color=BLUE, line=dict(width=2, color="white")),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y}} rows per {label}<extra></extra>",
    ))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title=f"rows / {label}", showgrid=True, gridcolor="rgba(128,128,128,0.15)", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return _base_layout(fig, height=300, title=f"{name} over time")


def correlation_heatmap(typed: pd.DataFrame, numeric_columns: list) -> go.Figure:
    """Pearson correlation matrix on a diverging blue↔red scale with a neutral midpoint."""
    if len(numeric_columns) < 2:
        return _empty("At least two numeric columns are needed for correlations")
    corr = typed[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, DIVERGING_NEG], [0.5, DIVERGING_MID], [1, DIVERGING_POS]],
        xgap=2, ygap=2, text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=10, title="r"),
    ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=max(300, 60 + 42 * len(numeric_columns)), title="Correlation matrix (Pearson)")


def status_breakdown(report: QualityReport) -> dict:
    """Counts by status for KPI display."""
    by_status = report.summary()["by_status"]
    return {s.value: by_status.get(s.value, 0) for s in CheckStatus}
