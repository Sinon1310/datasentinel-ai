"""Data Quality: score, check outcomes, detailed results, evidence and exports."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.dataset import AnalysisState
from app.components.ui import card, status_text, Kpi, badge_markdown, empty_state, error_state, fmt_int, fmt_score, kpi_row, \
    page_header, section
from datasentinel.pipeline import AnalysisResult
from datasentinel.quality.models import CheckStatus, Severity

# Generic, static guidance per rule type (UI copy, not data; never presented as a finding).
GUIDANCE = {
    "completeness.missing_values": "Trace why values are missing upstream; impute only where business rules allow.",
    "uniqueness.duplicate_records": "Deduplicate exact copies and check for re-run or double-loaded extracts.",
    "validity.data_types": "Fix or quarantine values that do not parse as the column's type at the source.",
    "validity.dates": "Correct impossible or out-of-range dates and agree on one canonical date format.",
    "accuracy.outliers_iqr": "Confirm extreme values are genuine before excluding or capping them.",
    "consistency.schema": "Align the data with its schema contract, or update the contract deliberately.",
}


def quality_kpis(result: AnalysisResult) -> list[Kpi]:
    """KPIs straight from ``QualityReport``. ``overall_score=None`` stays unavailable and grade stays 'N/A'."""
    quality = result.quality
    summary = quality.summary()
    critical = sum(1 for r in quality.results if r.status == CheckStatus.FAILED and r.severity == Severity.CRITICAL)
    unscored = quality.overall_score is None
    evaluated = sum(1 for d in quality.dimension_scores if d.score is not None)
    return [
        Kpi("Quality score", fmt_score(quality.overall_score),
            help="No check was evaluated, so there is no score." if unscored
            else "Weighted mean of dimension scores (see the score breakdown).",
            caption="No checks evaluated" if unscored else f"Weighted across {evaluated} dimensions"),
        Kpi("Grade", quality.grade, caption="A ≥ 90 · B ≥ 80 · C ≥ 70 · D ≥ 60"),
        Kpi("Passed checks", fmt_int(summary["by_status"]["passed"]), caption=f"of {summary['total_checks']} checks"),
        Kpi("Failed checks", fmt_int(summary["by_status"]["failed"]),
            help=f"{summary['by_status']['skipped']} check(s) skipped",
            caption=f"{summary['by_status']['skipped']} skipped"),
        Kpi("Critical issues", fmt_int(critical), help="Failed checks with critical severity",
            caption="Failed with critical severity"),
    ]


def results_table(result: AnalysisResult) -> pd.DataFrame:
    """Rule results with display columns, built from ``RuleResult`` fields."""
    rows = [{
        "check": r.rule_name, "column": r.target, "severity": r.severity.value, "status": r.status.value,
        "affected_rows": r.affected_rows, "affected_pct": round(r.affected_pct, 2), "description": r.message,
        "guidance": GUIDANCE.get(r.rule_id, "") if r.status == CheckStatus.FAILED else "",
        "dimension": r.dimension.value, "rule_id": r.rule_id,
    } for r in result.quality.results]
    return pd.DataFrame(rows)


def render(state: AnalysisState) -> None:
    """Render the Data Quality section."""
    result = state.result
    context = [f"Dataset · {state.dataset.filename}"] if state.dataset else []
    if result:
        context.append(f"Grade · {result.quality.grade}")
        context.append(f"{len(result.quality.results)} checks evaluated or skipped")
    def _downloads() -> None:
        st.download_button("Report (JSON)", result.quality.to_json(), icon=":material/download:", key="q_dl_json_head",
                           file_name=f"{result.dataset.name}.quality.json", mime="application/json")
        st.download_button("Results (CSV)", result.quality.results_dataframe().to_csv(index=False), key="q_dl_csv_head",
                           icon=":material/download:", file_name=f"{result.dataset.name}.quality.csv", mime="text/csv")

    page_header("Data Quality", "Deterministic rule-based validation with a transparent, reproducible score.",
                eyebrow="Analysis", context=context, actions=_downloads if result else None)
    if state.error_title:
        error_state(state.error_title, state.error_detail or "")
        return
    if result is None:
        empty_state("No quality checks yet", "Load a dataset from the sidebar to run the quality engine.",
                    why="Six deterministic rules score completeness, uniqueness, validity, consistency and accuracy.",
                    icon="check")
        return

    quality = result.quality
    kpi_row(quality_kpis(result))

    left, right = st.columns(2, gap="medium")
    with left:
        with card("q_1"):
            st.plotly_chart(charts.check_status_chart(quality), key="q_status_chart")
        with card("q_2"):
            st.plotly_chart(charts.missing_values_chart(result.profile), key="q_missing")
    with right:
        with card("q_3"):
            st.plotly_chart(charts.severity_chart(quality), key="q_sev")
        with card("q_4"):
            st.plotly_chart(charts.duplicates_chart(result.profile.n_rows, result.profile.duplicate_rows), key="q_dup")

    with card("q_5"):
        section("Score breakdown", "Each dimension's contribution sums to the overall score.")
        left, right = st.columns([3, 2], gap="medium")
        with left:
            st.dataframe(pd.DataFrame([{
                "dimension": d.dimension.value, "score": d.score,
                "weight": f"{d.effective_weight * 100:.0f}%" if d.score is not None else "n/a",
                "contribution": d.contribution, "checks": d.checks_evaluated, "failed": d.checks_failed,
            } for d in quality.dimension_scores]), hide_index=True)
        with right:
            m = quality.methodology
            st.markdown(
                "**How the score is calculated**\n\n"
                f"1. Check score = 100 × (1 − severity penalty). Passed = 100; skipped checks are excluded. "
                f"Penalties: `{m['severity_penalties']}`\n"
                "2. Dimension score = mean of its check scores.\n"
                f"3. Overall = weighted mean of dimension scores. Base weights `{m['dimension_weights']}`, "
                "re-normalised over evaluated dimensions.\n"
                "4. Grade: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else F. No evaluated checks ⇒ no score, grade N/A."
            )

    _render_results(result)



def _render_results(result: AnalysisResult) -> None:
    quality = result.quality
    df = results_table(result)
    with card("q_6"):
        section("Rule results", "Filter by outcome, severity or dimension.")
        f1, f2, f3 = st.columns(3)
        statuses = f1.multiselect("Status", sorted(df["status"].unique()), key="q_status",
                                  default=["failed"] if "failed" in df["status"].values else sorted(df["status"].unique()))
        severities = f2.multiselect("Severity", sorted(df["severity"].unique()), default=sorted(df["severity"].unique()),
                                    key="q_sev_filter")
        dims = f3.multiselect("Dimension", sorted(df["dimension"].unique()), default=sorted(df["dimension"].unique()),
                              key="q_dim")
        view = df[df["status"].isin(statuses) & df["severity"].isin(severities) & df["dimension"].isin(dims)]
        view = view.assign(status=view["status"].map(status_text), severity=view["severity"].str.capitalize())
        st.dataframe(
            view.drop(columns=["rule_id", "dimension"]), hide_index=True,
            column_config={
                "check": st.column_config.TextColumn("Check", width="medium"),
                "column": st.column_config.TextColumn("Column"),
                "severity": st.column_config.TextColumn("Severity"),
                "status": st.column_config.TextColumn("Status"),
                "affected_rows": st.column_config.NumberColumn("Affected rows", format="localized"),
                "affected_pct": st.column_config.NumberColumn("Affected %", format="%.2f"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "guidance": st.column_config.TextColumn("Suggested action (generic)", width="large"),
            },
        )
        st.caption(f"Showing {len(view)} of {len(df)} results.")

    failed = quality.failed_results
    if not failed:
        return
    with card("q_7"):
        section("Failed check details", "Evidence recorded by the quality engine for one failed check.")
        labels = {i: f"[{r.severity.value.upper()}] {r.rule_name} · {r.target}" for i, r in enumerate(failed)}
        choice = st.selectbox("Failed check", list(labels), format_func=labels.get, key="q_failed_pick")
        r = failed[choice]
        st.markdown(f"{badge_markdown(r.severity.value.capitalize(), r.severity.value)} {r.message}")
        if r.sample_values:
            st.markdown("**Sample offending values:** " + ", ".join(f"`{v}`" for v in r.sample_values))
        if r.rule_id in GUIDANCE:
            st.caption(f"Suggested action (generic): {GUIDANCE[r.rule_id]}")
        if r.column and r.column in result.profile.columns:
            column = result.schema.column(r.column)
            prof = result.profile.columns[r.column]
            st.markdown(f"**Column `{r.column}`** · type {column.semantic_type.value} ({column.storage_type}) · "
                        f"{prof.null_pct:.2f}% missing · {prof.unique_count:,} distinct")
        with st.expander("Raw evidence (details)"):
            st.json(json.loads(json.dumps(r.details, default=str)), expanded=False)
