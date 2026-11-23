"""Data Profiling: statistics per column, distributions, correlations and a data preview."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.dataset import AnalysisState
from app.components.ui import card, fmt_number, display_value, Kpi, empty_state, error_state, fmt_int, kpi_row, page_header
from datasentinel.pipeline import AnalysisResult
from datasentinel.schema.models import SemanticType

HIGH_NULL_PCT = 20.0  # display threshold for the "High-null columns" KPI


def profiling_kpis(result: AnalysisResult) -> list[Kpi]:
    schema, profile = result.schema, result.profile
    high_null = [p.name for p in profile.columns.values() if p.null_pct > HIGH_NULL_PCT]
    return [
        Kpi("Rows", fmt_int(profile.n_rows), caption=f"{profile.duplicate_rows:,} duplicates"),
        Kpi("Columns", fmt_int(profile.n_columns), caption=f"{profile.missing_cells_pct:.2f}% cells missing"),
        Kpi("Numeric", fmt_int(len(schema.columns_of_type(SemanticType.NUMERIC))), caption="statistics & distributions"),
        Kpi("Categorical / text", fmt_int(len(schema.columns_of_type(SemanticType.CATEGORICAL, SemanticType.TEXT))),
            caption="frequencies & cardinality"),
        Kpi("Date / time", fmt_int(len(schema.columns_of_type(SemanticType.DATETIME))), caption="ranges & trends"),
        Kpi("High-null columns", fmt_int(len(high_null)),
            help=f"Columns with more than {HIGH_NULL_PCT:.0f}% missing" + (f": {', '.join(high_null)}" if high_null else ""),
            caption=f"over {HIGH_NULL_PCT:.0f}% missing"),
    ]


def render(state: AnalysisState) -> None:
    """Render the Data Profiling section."""
    result = state.result
    page_header("Data Profiling", "Statistical profile of every column, computed once by the engine.",
                eyebrow="Analysis", context=[f"Dataset · {state.dataset.filename}"] if state.dataset else [])
    if state.error_title:
        error_state(state.error_title, state.error_detail or "")
        return
    if result is None:
        empty_state("Nothing to profile yet", "Load a dataset from the sidebar to see its statistical profile.",
                    why="Every column is profiled once by the engine; switching sections never recomputes it.",
                    icon="database")
        return

    kpi_row(profiling_kpis(result))
    tabs = st.tabs(["Column profile", "Statistics table", "Correlations", "Data preview"])
    with tabs[0]:
        _render_column(result)
    with tabs[1]:
        stats = result.profile.summary_dataframe()
        st.dataframe(stats.astype(object).map(display_value), hide_index=True)  # '—' instead of 'None' for n/a
    with tabs[2]:
        numeric = result.schema.columns_of_type(SemanticType.NUMERIC)
        with card("prof_1"):
            st.plotly_chart(charts.correlation_heatmap(result.typed_dataframe, numeric), key="prof_corr")
    with tabs[3]:
        _render_preview(result)


def _render_column(result: AnalysisResult) -> None:
    profile, typed = result.profile, result.typed_dataframe
    column = st.selectbox("Column", list(profile.columns), key="prof_col")
    p = profile.columns[column]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semantic type", p.semantic_type.value, border=True)
    c2.metric("Missing", f"{p.null_pct:.2f}%", help=f"{p.null_count:,} rows", border=True)
    c3.metric("Distinct", fmt_int(p.unique_count), border=True)
    c4.metric("Unique share", f"{p.unique_pct:.1f}%", help="Distinct values ÷ non-missing values", border=True)

    if p.numeric:
        n = p.numeric
        kpi_row([Kpi(label, fmt_number(value))
                 for label, value in [("Mean", n.mean), ("Median", n.median), ("Std dev", n.std), ("Min", n.min),
                                      ("Q1", n.q1), ("Q3", n.q3), ("Max", n.max)]])
        rule = next((r for r in result.quality.results if r.rule_id == "accuracy.outliers_iqr"
                     and r.column == column and r.details.get("lower_fence") is not None), None)
        with card("prof_2"):
            st.plotly_chart(charts.histogram_chart(typed[column], column,
                                                   rule.details["lower_fence"] if rule else None,
                                                   rule.details["upper_fence"] if rule else None), key="prof_hist")
        with card("prof_3"):
            st.plotly_chart(charts.box_chart(typed[column], column), key="prof_box")
        st.caption(f"Skewness {'—' if n.skewness is None else f'{n.skewness:.3f}'} · kurtosis "
                   f"{'—' if n.kurtosis is None else f'{n.kurtosis:.3f}'} · zeros {n.zeros:,} · negatives {n.negatives:,}"
                   + (f" · {n.infinite:,} infinite values excluded" if n.infinite else ""))
    elif p.categorical:
        cs = p.categorical
        kpi_row([
            Kpi("Unique values", fmt_int(cs.cardinality)),
            Kpi("Mode share", f"{cs.mode_pct:.1f}%", help="Share of the most frequent value (imbalance)"),
            Kpi("Normalised entropy", "—" if cs.normalized_entropy is None else f"{cs.normalized_entropy:.2f}",
                help="0 = one value dominates, 1 = perfectly even"),
            Kpi("Value length", f"{cs.mean_length}", caption=f"min {cs.min_length} · max {cs.max_length} characters"),
        ])
        with card("prof_4"):
            st.plotly_chart(charts.top_values_chart(cs, column), key="prof_top")
        if cs.other_values_count:
            st.caption(f"{cs.other_values_count:,} rows hold values outside the top {len(cs.top_values)}.")
    elif p.datetime:
        d = p.datetime
        kpi_row([Kpi("Earliest", str(d.min)[:10], caption=str(d.min)[11:19] or None),
                 Kpi("Latest", str(d.max)[:10], caption=str(d.max)[11:19] or None),
                 Kpi("Range", f"{d.span_days:,.1f} days"), Kpi("Distinct dates", fmt_int(d.distinct_dates))])
        with card("prof_5"):
            st.plotly_chart(charts.datetime_trend_chart(typed[column], column), key="prof_dt")
    elif p.boolean:
        b = p.boolean
        kpi_row([Kpi("True", fmt_int(b.true_count)), Kpi("False", fmt_int(b.false_count)),
                 Kpi("True share", f"{b.true_pct:.1f}%")])
    else:
        st.info("This column has no values to profile.")


def _render_preview(result: AnalysisResult) -> None:
    raw = result.raw_dataframe
    c1, c2, c3 = st.columns([1, 1, 2])
    view = c1.radio("View", ["Raw (as loaded)", "Typed (after casting)"], key="prev_view")
    n_rows = c2.number_input("Rows", min_value=5, max_value=max(5, len(raw)), value=min(100, len(raw)), step=25,
                             key="prev_rows")
    issue_filter = c3.selectbox("Show", ["All rows", "Rows with missing values", "Duplicate rows (all copies)"],
                                key="prev_filter")
    df = raw if view.startswith("Raw") else result.typed_dataframe
    if issue_filter == "Rows with missing values":
        df = df[result.typed_dataframe.isna().any(axis=1)]
    elif issue_filter.startswith("Duplicate"):
        df = df[raw.duplicated(keep=False)]
    columns = st.multiselect("Columns", list(df.columns), default=list(df.columns), key="prev_cols")
    st.caption(f"Showing {min(int(n_rows), len(df)):,} of {len(df):,} matching rows")
    st.dataframe(df[columns].head(int(n_rows)))
