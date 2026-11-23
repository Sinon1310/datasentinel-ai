"""Pipeline Observability page. Depends only on the typed contracts in ``app.observability``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.styles import BLUE, STATUS_COLORS, VIOLET
from app.components.ui import card, display_value, status_text, Kpi, badge_markdown, demo_banner, empty_state, fmt_duration_ms, fmt_int, fmt_pct, \
    kpi_row, page_header, section
from app.observability.contracts import (
    ObservabilityCommands,
    ObservabilityDataSource,
    RunDetail,
    RunFilters,
    RunStatus,
)
from app.observability.demo import DemoBackend
from app.observability.unavailable import NOT_CONNECTED, UnavailableBackend

WINDOWS = {"Last 24 hours": timedelta(hours=24), "Last 7 days": timedelta(days=7), "Last 30 days": timedelta(days=30),
           "All time": None}
_SOURCES = {"live": "Live data", "demo": "Synthetic demo"}


def get_backend(kind: str) -> ObservabilityDataSource:
    """Data source for the selected kind. The demo backend is per browser session and in memory only."""
    if kind == "demo":
        if "obs_demo_backend" not in st.session_state:
            st.session_state["obs_demo_backend"] = DemoBackend()
        return st.session_state["obs_demo_backend"]
    return UnavailableBackend()


def _use_demo() -> None:
    st.session_state["obs_source"] = "demo"


def _fmt_time(value: datetime | None) -> str:
    return "—" if value is None else value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _source_label(kind: str, live_connected: bool) -> str:
    if kind == "live":
        return "Live data" if live_connected else "Live data (not connected)"
    return "Synthetic demo"


def _render_connection_status(backend: ObservabilityDataSource, live_connected: bool) -> None:
    """Backend availability and the data being shown are separate facts; never imply a connection."""
    with st.container(horizontal=True, gap="small"):
        if live_connected:
            st.badge("Observability backend: connected", icon=":material/cloud_done:", color="green")
        else:
            st.badge("Observability backend: not connected", icon=":material/cloud_off:", color="orange")
        if backend.kind == "demo":
            st.badge("Showing: synthetic demo data (in memory)", icon=":material/science:", color="violet")
        elif backend.is_connected:
            st.badge("Showing: live data", icon=":material/sensors:", color="blue")
        else:
            st.badge("Showing: no data", icon=":material/block:", color="gray")
        st.badge("Environment: local workspace", color="gray")
        stamp = backend.last_refreshed_at()
        if stamp is not None:
            prefix = "Generated" if backend.kind == "demo" else "Refreshed"
            st.badge(f"{prefix}: {_fmt_time(stamp)}", icon=":material/schedule:", color="gray")


def render() -> None:
    """Render the Pipeline Observability page."""
    st.session_state.setdefault("obs_source", "live")
    live_connected = get_backend("live").is_connected
    page_header(
        "Pipeline Observability",
        "Execution history, health, failure classification and retry lineage for instrumented pipelines.",
        eyebrow="Operations",
    )
    st.segmented_control("Data source", list(_SOURCES), format_func=lambda k: _source_label(k, live_connected),
                         key="obs_source",
                         help="Live data appears once the observability backend is connected. "
                              "Synthetic demo data is generated in memory and never persisted.")
    kind = st.session_state.get("obs_source") or "live"
    backend = get_backend(kind)
    _render_connection_status(backend, live_connected)

    if not backend.is_connected:
        empty_state("No live pipeline data yet", NOT_CONNECTED, icon="cloud_off", tone="amber",
                    why="Execution history, failure classification and retry lineage need instrumented pipelines "
                        "reporting to the observability backend.",
                    steps=["Instrument a pipeline with the Python tracker (coming in Milestone 3).",
                           "Runs, tasks, failures and retries will appear here automatically.",
                           "Meanwhile, explore the page with clearly labelled synthetic data."])
        st.button("Explore synthetic demo data", key="obs_try_demo", on_click=_use_demo, icon=":material/science:")
        return

    if backend.kind == "demo":
        demo_banner("Generated deterministically in this browser session for demonstration. Pipeline ids start with "
                    "“demo.”; nothing is stored in the observability database and actions change only this session.")

    filters = _render_filters(backend)
    _render_kpis(backend, filters)
    _render_health(backend, filters)
    _render_trends(backend, filters)
    _render_failures(backend, filters)
    run_id = _render_recent(backend, filters)
    if run_id:
        detail = backend.get_run(run_id)
        if detail is not None:
            _render_detail(detail, backend if isinstance(backend, ObservabilityCommands) else None)


# ---------------------------------------------------------------------- sections


def _render_filters(backend: ObservabilityDataSource) -> RunFilters:
    pipelines = {p.pipeline_id: p.display_name for p in backend.list_pipelines()}
    with card("obs_1"):
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 3])
        chosen = c1.multiselect("Pipeline", list(pipelines), format_func=pipelines.get, key="obs_pipelines",
                                placeholder="All pipelines")
        statuses = c2.multiselect("Status", [s.value for s in RunStatus], key="obs_statuses", placeholder="All statuses")
        window = c3.selectbox("Time range", list(WINDOWS), index=2, key="obs_window")
        triggers = c4.multiselect("Trigger", backend.trigger_types(), key="obs_triggers", placeholder="All triggers")
        scope = c5.multiselect("Data", ["Real", "Synthetic"], default=["Real", "Synthetic"], key="obs_scope")
        stale_only = st.toggle("Only stale running executions", key="obs_stale",
                               help="RUNNING executions whose last heartbeat is older than the stale threshold. "
                                    "Stale is a display state; the stored status is not changed.")
    now = backend.last_refreshed_at() or datetime.now(timezone.utc)
    span = WINDOWS[window]
    return RunFilters(pipeline_ids=tuple(chosen), statuses=tuple(RunStatus(s) for s in statuses),
                      trigger_types=tuple(triggers), start=(now - span) if span else None,
                      end=now + timedelta(hours=1) if span else None,
                      include_real="Real" in scope, include_synthetic="Synthetic" in scope, stale_only=stale_only)


def _render_kpis(backend: ObservabilityDataSource, filters: RunFilters) -> None:
    s = backend.summary(filters)
    kpi_row([
        Kpi("Total executions", fmt_int(s.total_runs), help="Executions created in the selected range",
            caption=f"{s.pending_count} pending · {s.cancelled_count} cancelled"),
        Kpi("Success rate", fmt_pct(s.success_rate_pct), help="SUCCESS ÷ finished executions (success, failed, cancelled)",
            caption=f"{s.success_count:,} succeeded"),
        Kpi("Failed executions", fmt_int(s.failed_count), caption="see failure intelligence below"),
        Kpi("Running executions", fmt_int(s.running_count), help="Stale = no heartbeat within the threshold (display only)",
            caption=f"{s.stale_count} stale"),
        Kpi("Average duration", fmt_duration_ms(s.avg_duration_ms), help="Finished executions with a recorded duration",
            caption=f"p95 {fmt_duration_ms(s.p95_duration_ms)}"),
        Kpi("Processed rows", fmt_int(s.total_input_rows), help="Sum of recorded input rows",
            caption="recorded input rows"),
    ])


def _render_health(backend: ObservabilityDataSource, filters: RunFilters) -> None:
    left, right = st.columns([3, 2], gap="medium")
    with left, card("obs_2"):
        counts = pd.DataFrame([{"pipeline": c.pipeline_name, "status": c.status.value, "count": c.count}
                               for c in backend.status_counts(filters)])
        st.plotly_chart(charts.status_by_pipeline_chart(counts if not counts.empty else
                                                        pd.DataFrame(columns=["pipeline", "status", "count"])),
                        key="obs_status_chart")
    with right, card("obs_3"):
        section("Pipeline health")
        health = backend.pipeline_health(filters)
        if not health:
            st.caption("No executions match the filters.")
        for h in health:
            last = h.last_status.value if h.last_status else "none"
            st.markdown(f"**{h.pipeline_name}** {badge_markdown(last.capitalize(), last)}  \n"
                        f"<span style='color:#64748B;font-size:0.85rem'>{h.runs} runs · success "
                        f"{fmt_pct(h.success_rate_pct)} · p95 {fmt_duration_ms(h.p95_duration_ms)}"
                        f"{' · last failure ' + h.last_failure_category.value.replace('_', ' ').lower() if h.last_failure_category else ''}"
                        "</span>", unsafe_allow_html=True)


def _render_trends(backend: ObservabilityDataSource, filters: RunFilters) -> None:
    points = backend.trends(filters)
    frame = pd.DataFrame([{
        "bucket": p.bucket, "success": p.success, "failed": p.failed, "cancelled": p.cancelled, "active": p.active,
        "avg_min": None if p.avg_duration_ms is None else p.avg_duration_ms / 60000,
        "p95_min": None if p.p95_duration_ms is None else p.p95_duration_ms / 60000,
        "throughput": p.avg_throughput_rows_per_s,
    } for p in points])
    empty = pd.DataFrame(columns=["bucket", "success", "failed", "cancelled", "active", "avg_min", "p95_min", "throughput"])
    frame = frame if not frame.empty else empty
    with card("obs_4"):
        st.plotly_chart(charts.run_volume_chart(frame), key="obs_volume")
    left, right = st.columns(2, gap="medium")
    with left, card("obs_5"):
        st.plotly_chart(charts.line_trend_chart(frame, [("avg_min", "Average", BLUE), ("p95_min", "p95", VIOLET)],
                                                "Execution duration", "Minutes", " min"), key="obs_duration")
    with right, card("obs_6"):
        st.plotly_chart(charts.line_trend_chart(frame, [("throughput", "Rows per second", STATUS_COLORS["success"])],
                                                "Throughput (successful executions)", "Rows / s"), key="obs_throughput")


def _render_failures(backend: ObservabilityDataSource, filters: RunFilters) -> None:
    distribution = backend.failure_distribution(filters)
    left, right = st.columns([2, 3], gap="medium")
    with left, card("obs_7"):
        st.plotly_chart(charts.failure_category_chart([d.category.value for d in distribution],
                                                      [d.count for d in distribution]), key="obs_failures")
    with right, card("obs_8"):
        section("Most common failure types", "Classified deterministically; messages are redacted before storage.")
        types = backend.top_failure_types(filters)
        if not types:
            st.caption("No failures in the selected range.")
        else:
            st.dataframe(pd.DataFrame([{"error type": t.error_type, "category": t.category.value, "count": t.count,
                                        "last seen": _fmt_time(t.last_seen)} for t in types]), hide_index=True)


def _render_recent(backend: ObservabilityDataSource, filters: RunFilters) -> str | None:
    page = backend.list_runs(filters, limit=200)
    with card("obs_9"):
        section("Recent executions", f"{page.total:,} matching executions (newest first, up to 200 shown).")
        if not page.items:
            empty_state("No executions match these filters", "Widen the time range or clear the pipeline, status or "
                        "trigger filters.", icon="search", tone="slate")
            return None
        frame = pd.DataFrame([{
            "run id": r.pipeline_run_id[:12], "pipeline": r.pipeline_name, "status": status_text(r.status.value) + (" · stale" if r.is_stale else ""),
            "trigger": r.trigger_type, "started (UTC)": _fmt_time(r.started_at), "duration": fmt_duration_ms(r.duration_ms),
            "input rows": display_value(r.input_rows), "output rows": display_value(r.output_rows), "attempt": r.attempt_number,
            "data": "synthetic" if r.is_synthetic else "real",
        } for r in page.items])
        ids = [r.pipeline_run_id for r in page.items]
        event = st.dataframe(frame, hide_index=True, on_select="rerun", selection_mode="single-row", key="obs_table",
                             column_config={"status": st.column_config.TextColumn("status", help="Persisted status; "
                                                                                  "'STALE' is a derived display state"),
                                            "input rows": st.column_config.TextColumn("input rows", alignment="right"),
                                            "output rows": st.column_config.TextColumn("output rows", alignment="right")})
        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            st.session_state["obs_run"] = ids[selected_rows[0]]
        labels = {r.pipeline_run_id: f"{r.pipeline_name} · {r.display_status} · {_fmt_time(r.created_at)} · "
                                     f"{r.pipeline_run_id[:8]}" for r in page.items}
        if st.session_state.get("obs_run") not in labels:
            st.session_state.pop("obs_run", None)
        return st.selectbox("Inspect execution", ids, format_func=labels.get, key="obs_run",
                            help="Or select a row in the table above.")


# ---------------------------------------------------------------------- run detail


def _render_detail(detail: RunDetail, commands: ObservabilityCommands | None) -> None:
    run, metrics = detail.run, detail.metrics
    with card("obs_10"):
        st.markdown(f"### {run.pipeline_name} {badge_markdown(run.display_status, 'stale' if run.is_stale else run.status.value)}"
                    + (f" {badge_markdown('Synthetic', 'info')}" if run.is_synthetic else ""))
        st.caption(f"Pipeline run {run.pipeline_run_id} · attempt {run.attempt_number} of {len(detail.attempts)} · "
                   f"trigger {run.trigger_type}" + (f" · requested by {detail.requested_by}" if detail.requested_by else ""))
        kpi_row([
            Kpi("Duration", fmt_duration_ms(run.duration_ms), help=f"Measurement source: {run.duration_source}"),
            Kpi("Input rows", fmt_int(run.input_rows)),
            Kpi("Output rows", fmt_int(run.output_rows)),
            Kpi("Throughput", "—" if not metrics or metrics.input_throughput_rows_per_s is None
                else f"{metrics.input_throughput_rows_per_s:,.0f} rows/s", help="Input rows ÷ duration"),
            Kpi("Failed tasks", fmt_int(metrics.failed_task_count) if metrics else "—",
                help=f"{metrics.task_retry_count} task retries" if metrics else "Available when the run finishes"),
        ])
        st.caption(f"Created {_fmt_time(run.created_at)} · started {_fmt_time(run.started_at)} · ended "
                   f"{_fmt_time(run.ended_at)}" + (f" · last heartbeat {_fmt_time(detail.heartbeat_at)}"
                                                   if run.status == RunStatus.RUNNING else ""))
        if run.is_stale:
            st.warning("This execution is RUNNING but has not sent a heartbeat within the stale threshold. "
                       "Its stored status is unchanged; check the worker that owns it.")

        _render_actions(detail, commands)

        tabs = st.tabs(["Tasks", "Failure", "Attempts", "Events", "Parameters"])
        with tabs[0]:
            tasks = pd.DataFrame([{"label": f"{t.task_name} #{t.task_attempt}", "start": t.started_at,
                                   "end": t.ended_at or (detail.heartbeat_at or t.started_at), "status": t.status.value}
                                  for t in detail.tasks])
            st.plotly_chart(charts.task_timeline_chart(tasks if not tasks.empty else
                                                       pd.DataFrame(columns=["label", "start", "end", "status"])),
                            key="obs_timeline")
            if detail.tasks:
                st.dataframe(pd.DataFrame([{"task": t.task_name, "attempt": t.task_attempt, "status": status_text(t.status.value),
                                            "started": _fmt_time(t.started_at), "duration": fmt_duration_ms(t.duration_ms),
                                            "input rows": display_value(t.input_rows), "output rows": display_value(t.output_rows)}
                                           for t in detail.tasks]), hide_index=True)
        with tabs[1]:
            if not detail.failures:
                st.caption("No failures recorded for this execution.")
            for failure in detail.failures:
                st.markdown(f"{badge_markdown(failure.category.value.replace('_', ' '), 'failed')} "
                            f"{'Primary failure' if failure.is_primary else 'Secondary failure'} · stage "
                            f"`{failure.stage or '—'}` · classified by `{failure.classifier_rule}` ({failure.category_source})")
                st.code(f"{failure.error_type}: {failure.error_message}", language=None)
                if failure.stack:
                    st.caption("Stack (module:function:line, innermost last — no source lines or local variables "
                               "are stored): " + " → ".join(failure.stack))
        with tabs[2]:
            st.dataframe(pd.DataFrame([{"attempt": a.attempt_number, "status": status_text(a.status.value), "trigger": a.trigger_type,
                                        "created": _fmt_time(a.created_at),
                                        "failure": a.failure_category.value if a.failure_category else "—",
                                        "pipeline run id": a.pipeline_run_id} for a in detail.attempts]), hide_index=True)
            st.caption("Each retry is a separate attempt linked to the previous one. Retries are requested by people "
                       "and never executed by DataSentinel.")
        with tabs[3]:
            st.dataframe(pd.DataFrame([{"at": _fmt_time(e.at), "event": e.event_type,
                                        "transition": f"{e.from_status or '—'} → {e.to_status}" if e.to_status else "",
                                        "actor": e.actor, "details": str(e.details) if e.details else ""}
                                       for e in detail.events]), hide_index=True)
        with tabs[4]:
            st.caption("Values are redacted before storage (secrets, credentials, PII).")
            st.json({"parameters": detail.parameters, "metadata": detail.metadata}, expanded=False)


def _render_actions(detail: RunDetail, commands: ObservabilityCommands | None) -> None:
    result = st.session_state.pop("obs_action_result", None)
    if result is not None:
        (st.success if result.ok else st.error)(result.message)
    allowed = detail.allowed_actions
    can_retry = commands is not None and allowed.can_request_retry
    can_cancel = commands is not None and allowed.can_cancel
    actions = st.container(horizontal=True, gap="small")
    if actions.button("Request retry", key="obs_retry_btn", icon=":material/replay:", disabled=not can_retry,
                 help=allowed.retry_blocked_reason or "Creates a PENDING attempt. Nothing is executed."):
        _retry_dialog(detail.run.pipeline_run_id, commands)
    if actions.button("Cancel pending run", key="obs_cancel_btn", icon=":material/block:", disabled=not can_cancel,
                 help=allowed.cancel_blocked_reason or "Marks this PENDING attempt as CANCELLED."):
        _cancel_dialog(detail.run.pipeline_run_id, commands)


@st.dialog("Request retry")
def _retry_dialog(pipeline_run_id: str, commands: ObservabilityCommands) -> None:
    st.markdown("This creates a new **PENDING** attempt linked to this execution. **Nothing is executed**: the owning "
                "pipeline must claim and run the attempt itself.")
    requester = st.text_input("Your name", key="obs_retry_who")
    reason = st.text_area("Reason", key="obs_retry_why", max_chars=500)
    if st.button("Confirm retry request", type="primary", key="obs_retry_confirm",
                 disabled=not (requester.strip() and reason.strip())):
        st.session_state["obs_action_result"] = commands.request_retry(pipeline_run_id, requester, reason)
        st.rerun()


@st.dialog("Cancel pending run")
def _cancel_dialog(pipeline_run_id: str, commands: ObservabilityCommands) -> None:
    st.markdown("This marks the **PENDING** attempt as **CANCELLED**. Running executions cannot be cancelled here.")
    actor = st.text_input("Your name", key="obs_cancel_who")
    reason = st.text_area("Reason", key="obs_cancel_why", max_chars=500)
    if st.button("Confirm cancellation", type="primary", key="obs_cancel_confirm",
                 disabled=not (actor.strip() and reason.strip())):
        st.session_state["obs_action_result"] = commands.cancel_pending(pipeline_run_id, actor, reason)
        st.rerun()
