"""Page sections. Each function renders one area of the dashboard from an AnalysisResult."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.styles import STATUS_ICONS
from datasentinel.pipeline import AnalysisResult
from datasentinel.quality.models import CheckStatus
from datasentinel.schema.contract import SchemaContract
from datasentinel.schema.models import SemanticType


def _fmt_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


# ---------------------------------------------------------------------- header


def render_kpis(result: AnalysisResult) -> None:
    """Headline KPI row shown above the tabs."""
    profile, quality = result.profile, result.quality
    summary = quality.summary()
    score = "—" if quality.overall_score is None else f"{quality.overall_score:.1f}"
    cols = st.columns(6)
    cols[0].metric("Quality score", score, help="Weighted mean of dimension scores (see Data Quality tab).",
                   border=True)
    cols[1].metric("Grade", quality.grade, border=True)
    cols[2].metric("Rows", f"{profile.n_rows:,}", border=True)
    cols[3].metric("Columns", f"{profile.n_columns:,}", border=True)
    cols[4].metric("Failed checks", f"{summary['by_status']['failed']} / {summary['total_checks']}", border=True)
    cols[5].metric("Missing cells", f"{profile.missing_cells_pct:.2f}%", border=True)


# ---------------------------------------------------------------------- overview


def render_overview(result: AnalysisResult) -> None:
    """Dataset overview tab."""
    meta, profile = result.dataset.metadata, result.profile
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Dataset")
        facts = {
            "File": meta.filename,
            "Format": meta.file_type.upper() + (f" ({meta.json_layout})" if meta.json_layout else ""),
            "File size": _fmt_bytes(meta.size_bytes),
            "In-memory size": _fmt_bytes(profile.memory_bytes),
            "Encoding": meta.encoding,
            "Delimiter": repr(meta.delimiter) if meta.delimiter else "—",
            "Duplicate rows": f"{profile.duplicate_rows:,} ({profile.duplicate_rows_pct:.2f}%)",
            "Missing cells": f"{profile.missing_cells:,} of {profile.total_cells:,}",
            "Primary key": result.schema.primary_key.label if result.schema.primary_key else "None detected",
            "Load time": f"{meta.load_seconds * 1000:.0f} ms",
        }
        st.dataframe(pd.DataFrame(facts.items(), columns=["Property", "Value"]), hide_index=True)
        if meta.warnings:
            with st.expander(f"⚠️ {len(meta.warnings)} ingestion warning(s)", expanded=True):
                for warning in meta.warnings:
                    st.markdown(f"- {warning}")
    with right:
        st.plotly_chart(charts.missing_values_chart(profile), key="ov_missing")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.type_distribution_chart(result.schema), key="ov_types")
        with c2:
            st.plotly_chart(charts.severity_chart(result.quality), key="ov_severity")

    st.subheader("Top issues")
    failed = result.quality.failed_results[:5]
    if not failed:
        st.success("All quality checks passed.")
    for r in failed:
        st.markdown(f"{STATUS_ICONS[r.severity.value]} **{r.severity.value.upper()}** · `{r.target}` · "
                    f"{r.rule_name} — {r.message}")


# ---------------------------------------------------------------------- schema


def render_schema(result: AnalysisResult) -> None:
    """Schema explorer tab."""
    schema = result.schema
    st.subheader("Primary key candidates")
    if schema.primary_key_candidates:
        st.dataframe(
            pd.DataFrame([{"key": pk.label, "type": "composite" if pk.is_composite else "single",
                           "confidence": pk.confidence, "evidence": "; ".join(pk.reasons)}
                          for pk in schema.primary_key_candidates]),
            hide_index=True,
            column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        )
    else:
        st.info("No column or column pair uniquely identifies every row (duplicates or nulls prevent it).")

    st.subheader("Columns")
    types = [t.value for t in SemanticType if t.value in schema.type_summary()]
    selected = st.multiselect("Filter by semantic type", types, default=types, key="schema_types")
    table = schema.to_dataframe()
    table = table[table["semantic_type"].isin(selected)]
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "parse_success_ratio": st.column_config.NumberColumn("parse ratio", format="%.3f"),
            "pk_candidate": st.column_config.CheckboxColumn("PK?"),
            "nullable": st.column_config.CheckboxColumn("nullable"),
        },
    )

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download schema report (JSON)", json.dumps(schema.to_dict(), indent=2, default=str),
                       file_name=f"{result.dataset.name}.schema.json", mime="application/json")
    contract = SchemaContract.from_schema_report(schema)
    c2.download_button("⬇️ Download as schema contract (JSON)", contract.to_json(),
                       file_name=f"{result.dataset.name}.contract.json", mime="application/json",
                       help="Edit this file and upload it in the sidebar to validate future loads against it.")


# ---------------------------------------------------------------------- quality


def render_quality(result: AnalysisResult) -> None:
    """Data quality tab: score breakdown, rule results and evidence."""
    quality = result.quality
    left, right = st.columns([1, 2], gap="large")
    with left:
        score = "—" if quality.overall_score is None else f"{quality.overall_score:.1f}"
        st.markdown(f'<div class="ds-muted">Overall quality score</div>'
                    f'<div><span class="ds-score">{score}</span><span class="ds-grade">/ 100 · Grade {quality.grade}'
                    f'</span></div>', unsafe_allow_html=True)
        counts = charts.status_breakdown(quality)
        st.markdown(f"{STATUS_ICONS['passed']} {counts['passed']} passed · {STATUS_ICONS['failed']} "
                    f"{counts['failed']} failed · {STATUS_ICONS['skipped']} {counts['skipped']} skipped")
        st.markdown("**Score breakdown**")
        st.dataframe(
            pd.DataFrame([{
                "dimension": d.dimension.value,
                "score": d.score,
                "weight": f"{d.effective_weight * 100:.0f}%" if d.score is not None else "n/a",
                "contribution": d.contribution,
                "checks": d.checks_evaluated,
                "failed": d.checks_failed,
            } for d in quality.dimension_scores]),
            hide_index=True,
        )
        with st.expander("How is the score calculated?"):
            m = quality.methodology
            st.markdown(
                f"1. **Check score** = 100 × (1 − severity penalty). Passed = 100; skipped checks are excluded. "
                f"Penalties: `{m['severity_penalties']}`\n"
                "2. **Dimension score** = mean of its check scores.\n"
                f"3. **Overall** = weighted mean of dimension scores. Base weights `{m['dimension_weights']}`, "
                "re-normalised over dimensions that had at least one evaluated check.\n"
                "4. **Grade**: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else F.\n\n"
                "The *contribution* column sums exactly to the overall score."
            )
    with right:
        st.plotly_chart(charts.dimension_scores_chart(quality), key="q_dims")
        st.plotly_chart(charts.severity_chart(quality), key="q_sev")

    st.subheader("Rule results")
    df = quality.results_dataframe()
    f1, f2, f3 = st.columns(3)
    statuses = f1.multiselect("Status", sorted(df["status"].unique()), default=["failed"]
                              if "failed" in df["status"].values else sorted(df["status"].unique()), key="q_status")
    severities = f2.multiselect("Severity", sorted(df["severity"].unique()), default=sorted(df["severity"].unique()),
                                key="q_sev_filter")
    dims = f3.multiselect("Dimension", sorted(df["dimension"].unique()), default=sorted(df["dimension"].unique()),
                          key="q_dim")
    view = df[df["status"].isin(statuses) & df["severity"].isin(severities) & df["dimension"].isin(dims)].copy()
    view.insert(0, "", view["status"].map(STATUS_ICONS))
    st.dataframe(view, hide_index=True,
                 column_config={"score": st.column_config.NumberColumn(format="%.0f"),
                                "affected_pct": st.column_config.NumberColumn("affected %", format="%.2f")})

    failed = quality.failed_results
    if failed:
        st.subheader("Evidence for failed checks")
        for r in failed:
            with st.expander(f"{STATUS_ICONS[r.severity.value]} [{r.severity.value.upper()}] {r.rule_name} · {r.target}"):
                st.markdown(r.message)
                if r.sample_values:
                    st.markdown("**Sample offending values:** " + ", ".join(f"`{v}`" for v in r.sample_values))
                st.json(json.loads(json.dumps(r.details, default=str)), expanded=False)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download quality report (JSON)", quality.to_json(),
                       file_name=f"{result.dataset.name}.quality.json", mime="application/json")
    c2.download_button("⬇️ Download rule results (CSV)", df.to_csv(index=False),
                       file_name=f"{result.dataset.name}.quality.csv", mime="text/csv")


# ---------------------------------------------------------------------- profiling


def render_profiling(result: AnalysisResult) -> None:
    """Column profiling tab with interactive charts."""
    profile, typed = result.profile, result.typed_dataframe
    st.subheader("Column statistics")
    st.dataframe(profile.summary_dataframe(), hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ("mean", "median", "std")})

    st.subheader("Column explorer")
    column = st.selectbox("Column", list(profile.columns), key="prof_col")
    p = profile.columns[column]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semantic type", p.semantic_type.value, border=True)
    c2.metric("Missing", f"{p.null_pct:.2f}%", border=True)
    c3.metric("Distinct", f"{p.unique_count:,}", border=True)
    c4.metric("Unique share", f"{p.unique_pct:.1f}%", border=True)

    if p.numeric:
        n = p.numeric
        outlier = next((r for r in result.quality.results
                        if r.rule_id == "accuracy.outliers_iqr" and r.column == column and r.details.get("lower_fence") is not None), None)
        lower = outlier.details["lower_fence"] if outlier else None
        upper = outlier.details["upper_fence"] if outlier else None
        st.plotly_chart(charts.histogram_chart(typed[column], column, lower, upper), key="prof_hist")
        st.plotly_chart(charts.box_chart(typed[column], column), key="prof_box")
        stats = {"min": n.min, "p05": n.p05, "q1": n.q1, "median": n.median, "mean": n.mean, "q3": n.q3,
                 "p95": n.p95, "max": n.max, "std": n.std, "iqr": n.iqr, "skewness": n.skewness,
                 "kurtosis": n.kurtosis, "zeros": n.zeros, "negatives": n.negatives}
        st.dataframe(pd.DataFrame([stats]), hide_index=True)
    elif p.categorical:
        cs = p.categorical
        st.plotly_chart(charts.top_values_chart(cs, column), key="prof_top")
        st.dataframe(pd.DataFrame([{
            "cardinality": cs.cardinality, "mode": cs.mode, "mode %": round(cs.mode_pct, 2),
            "normalized entropy": cs.normalized_entropy, "other values (not in top)": cs.other_values_count,
            "min length": cs.min_length, "mean length": cs.mean_length, "max length": cs.max_length,
        }]), hide_index=True)
    elif p.datetime:
        st.plotly_chart(charts.datetime_trend_chart(typed[column], column), key="prof_dt")
        st.dataframe(pd.DataFrame([{"min": str(p.datetime.min), "max": str(p.datetime.max),
                                    "span (days)": p.datetime.span_days,
                                    "distinct dates": p.datetime.distinct_dates}]), hide_index=True)
    elif p.boolean:
        b = p.boolean
        st.dataframe(pd.DataFrame([{"true": b.true_count, "false": b.false_count, "true %": round(b.true_pct, 2)}]),
                     hide_index=True)
    else:
        st.info("This column has no values to profile.")

    numeric_cols = result.schema.columns_of_type(SemanticType.NUMERIC)
    if len(numeric_cols) >= 2:
        st.subheader("Correlations")
        st.plotly_chart(charts.correlation_heatmap(typed, numeric_cols), key="prof_corr")


# ---------------------------------------------------------------------- preview


def render_preview(result: AnalysisResult) -> None:
    """Data preview tab with raw / typed views and issue filters."""
    c1, c2, c3 = st.columns([1, 1, 2])
    view = c1.radio("View", ["Raw (as loaded)", "Typed (after casting)"], key="prev_view")
    n_rows = c2.number_input("Rows", min_value=5, max_value=max(5, len(result.raw_dataframe)), value=min(100, len(result.raw_dataframe)),
                             step=25, key="prev_rows")
    issue_filter = c3.selectbox("Show", ["All rows", "Rows with missing values", "Duplicate rows (all copies)"],
                                key="prev_filter")
    raw = result.raw_dataframe
    df = raw if view.startswith("Raw") else result.typed_dataframe
    if issue_filter == "Rows with missing values":
        df = df[result.typed_dataframe.isna().any(axis=1)]
    elif issue_filter.startswith("Duplicate"):
        df = df[raw.duplicated(keep=False)]
    columns = st.multiselect("Columns", list(df.columns), default=list(df.columns), key="prev_cols")
    st.caption(f"Showing {min(int(n_rows), len(df)):,} of {len(df):,} matching rows")
    st.dataframe(df[columns].head(int(n_rows)))


def render_welcome() -> None:
    """Landing state before any dataset is loaded."""
    st.info("Upload a CSV or JSON file in the sidebar, or load the bundled sample dataset to get started.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 🧬 Schema inference\nSemantic types, primary-key detection and exportable schema contracts.")
    c2.markdown("#### 📊 Profiling\nNulls, cardinality, distributions, quantiles and correlations.")
    c3.markdown("#### 🛡️ Quality engine\nDeterministic rules, severity levels and a transparent, reproducible score.")


def failed_count(result: AnalysisResult) -> int:
    """Number of failed checks (used for the tab label)."""
    return sum(1 for r in result.quality.results if r.status == CheckStatus.FAILED)
