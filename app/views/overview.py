"""Overview: session summary, headline KPIs, top issues and next actions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.dataset import AnalysisState
from app.components.shell import MAIN_PAGE
from app.components.ui import card, Kpi, badge_markdown, empty_state, error_state, fmt_bytes, fmt_int, fmt_score, \
    kpi_row, page_header, section
from datasentinel.config import get_settings
from datasentinel.investigation.workspace import RunStore
from datasentinel.pipeline import AnalysisResult
from datasentinel.quality.models import CheckStatus, Severity


def overview_kpis(result: AnalysisResult) -> list[Kpi]:
    """Headline KPIs, all taken from engine results."""
    profile, quality = result.profile, result.quality
    failed = sum(1 for r in quality.results if r.status == CheckStatus.FAILED)
    meta = result.dataset.metadata
    types = sorted(result.schema.type_summary().items(), key=lambda kv: -kv[1])[:2]
    by_sev = quality.summary()["failed_by_severity"]
    severity_mix = " · ".join(f"{n} {s}" for s, n in by_sev.items() if n) or "none failed"
    return [
        Kpi("Rows", fmt_int(profile.n_rows), caption=f"{meta.file_type.upper()} · {fmt_bytes(meta.size_bytes)}"),
        Kpi("Columns", fmt_int(profile.n_columns), caption=" · ".join(f"{n} {t}" for t, n in types)),
        Kpi("Quality score", fmt_score(quality.overall_score),
            help="Weighted mean of dimension scores; '—' means no check was evaluated.",
            caption=f"Grade {quality.grade}" if quality.overall_score is not None else "No checks evaluated"),
        Kpi("Missing cells", fmt_int(profile.missing_cells), caption=f"{profile.missing_cells_pct:.2f}% of all cells"),
        Kpi("Duplicate rows", fmt_int(profile.duplicate_rows), caption=f"{profile.duplicate_rows_pct:.2f}% of rows"),
        Kpi("Detected issues", fmt_int(failed), help="Number of failed quality checks", caption=severity_mix),
    ]


def next_actions(result: AnalysisResult, has_baseline: bool, ai_configured: bool) -> list[str]:
    """Deterministic next-step suggestions derived from the analysis (no LLM)."""
    quality = result.quality
    severe = [r for r in quality.failed_results if r.severity in (Severity.CRITICAL, Severity.HIGH)]
    actions = []
    if severe:
        actions.append(f"Review {len(severe)} high or critical quality issue(s) in Data Quality.")
    elif quality.failed_results:
        actions.append(f"Review {len(quality.failed_results)} lower-severity issue(s) in Data Quality.")
    else:
        actions.append("All quality checks passed. Export the schema contract to lock in today's structure.")
    if not has_baseline:
        actions.append("Upload a schema baseline (contract or exported schema) to enable drift detection.")
    if quality.failed_results:
        actions.append("Run an AI investigation to trace root causes and propose reviewable fixes."
                       if ai_configured else "Configure GROQ_API_KEY to enable AI investigation of these issues.")
    return actions


def _recent_investigations() -> pd.DataFrame | None:
    """Persisted investigation runs (real records only), without creating directories as a side effect."""
    settings = get_settings()
    if not (settings.data_dir / "runs").is_dir():
        return None
    runs = RunStore(settings.data_dir).list_runs()[:5]
    if not runs:
        return None
    return pd.DataFrame([{"created (UTC)": m.created_at, "dataset": m.filename, "privacy": m.privacy_mode,
                          "artefacts": len(m.artifacts)} for m in runs])


def render(state: AnalysisState, has_baseline: bool, ai_configured: bool) -> None:
    """Render the Overview section."""
    result = state.result
    context = []
    if state.dataset:
        context.append(f"Dataset · {state.dataset.filename}")
    if result:
        context.append(f"Analysed · {result.quality.generated_at.replace('T', ' ').replace('+00:00', ' UTC')}")
    page_header("Overview", "Profile datasets, validate quality deterministically, investigate issues with "
                "evidence-grounded AI and monitor pipeline health.", eyebrow="DataSentinel AI", context=context)

    if state.error_title:
        error_state(state.error_title, state.error_detail or "")
        return
    if result is None:
        _render_empty()
        return

    kpi_row(overview_kpis(result))
    left, right = st.columns(2, gap="medium")
    with left:
        with card("ov_1"):
            st.plotly_chart(charts.dimension_scores_chart(result.quality), key="ov_dims")
        with card("ov_2"):
            st.plotly_chart(charts.missing_values_chart(result.profile), key="ov_missing")
    with right:
        with card("ov_3"):
            st.plotly_chart(charts.severity_chart(result.quality), key="ov_severity")
        with card("ov_4"):
            st.plotly_chart(charts.type_distribution_chart(result.schema), key="ov_types")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with card("ov_5"):
            section("Top quality issues", "Most severe failed checks first.")
            failed = result.quality.failed_results[:5]
            if not failed:
                st.success("All quality checks passed.")
            for r in failed:
                st.markdown(f"{badge_markdown(r.severity.value.capitalize(), r.severity.value)} "
                            f"**{r.rule_name}** · `{r.target}` — {r.message}")
            st.page_link(MAIN_PAGE, label="Open Data Quality", icon=":material/arrow_forward:",
                         query_params={"section": "quality"})
    with right:
        with card("ov_6"):
            section("Recommended next actions")
            for action in next_actions(result, has_baseline, ai_configured):
                st.markdown(f"- {action}")
            st.page_link(MAIN_PAGE, label="Open AI Investigation", icon=":material/arrow_forward:",
                         query_params={"section": "investigation"})

    with card("ov_7"):
        section("Dataset summary")
        meta, profile = result.dataset.metadata, result.profile
        facts = {
            "File": meta.filename,
            "Format": meta.file_type.upper() + (f" ({meta.json_layout})" if meta.json_layout else ""),
            "File size": fmt_bytes(meta.size_bytes),
            "In-memory size": fmt_bytes(profile.memory_bytes),
            "Encoding": meta.encoding,
            "Delimiter": repr(meta.delimiter) if meta.delimiter else "—",
            "Primary key": result.schema.primary_key.label if result.schema.primary_key else "None detected",
            "Schema baseline": "Provided" if has_baseline else "Not provided",
            "Load time": f"{meta.load_seconds * 1000:.0f} ms",
        }
        st.dataframe(pd.DataFrame(facts.items(), columns=["Property", "Value"]), hide_index=True)
        if meta.warnings:
            with st.expander(f"{len(meta.warnings)} ingestion warning(s)", expanded=True):
                for warning in meta.warnings:
                    st.markdown(f"- {warning}")

    recent = _recent_investigations()
    if recent is not None:
        with card("ov_8"):
            section("Recent AI investigations", "Persisted investigation runs on this machine.")
            st.dataframe(recent, hide_index=True)


def _render_empty() -> None:
    # Primary call to action first; the card explains what happens next without repeating it.
    st.info("Upload a CSV or JSON file in the sidebar, or load a bundled sample dataset to get started.",
            icon=":material/upload_file:")
    empty_state(
        "No dataset loaded",
        "Load a dataset to profile its columns, validate quality and score it.",
        why="Schema, profile and quality checks run locally and deterministically. No data leaves your machine "
            "unless you start an AI investigation.",
        steps=["Review the quality score and the issues it found.",
               "Optionally run an evidence-grounded AI investigation and approve fixes."],
        icon="database",
    )
    cols = st.columns(3, gap="medium")
    blurbs = [
        ("Data quality", "Missing values, duplicates, invalid types and dates, IQR outliers and schema checks with "
                         "a transparent, reproducible score."),
        ("AI investigation", "Agents explain issues using cited evidence, separate facts from hypotheses and propose "
                             "fixes that run only after human approval."),
        ("Pipeline observability", "Run history, failure classification, duration trends and retry lineage for "
                                   "instrumented pipelines."),
    ]
    for col, (title, body) in zip(cols, blurbs):
        with card(f"ov_blurb_{title.split()[0].lower()}", parent=col):
            st.markdown(f"**{title}**")
            st.caption(body)
    recent = _recent_investigations()
    if recent is not None:
        with card("ov_10"):
            section("Recent AI investigations", "Persisted investigation runs on this machine.")
            st.dataframe(recent, hide_index=True)
