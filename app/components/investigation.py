"""AI Investigation tab: start runs, follow progress, review evidence and approve fixes.

The UI never calls an LLM or executes SQL itself. It talks only to
:class:`datasentinel_agents.service.InvestigationService`. Dataset-derived tokens are
rehydrated to real values here, for local display only.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from datasentinel.config import get_settings
from datasentinel.exceptions import DataSentinelError
from datasentinel.investigation.workspace import RunStoreError
from datasentinel_agents.config import get_agent_settings
from datasentinel_agents.llm import LLMConfigurationError
from datasentinel_agents.service import InvestigationService, RunSnapshot
from datasentinel_agents.services import Services

NODE_LABELS = {
    "prepare_context": "Preparing deterministic context",
    "data_investigator": "Data Investigator",
    "schema_analyst": "Schema Analyst",
    "grounding_check": "Grounding claims against evidence",
    "root_cause_analyst": "Root Cause Analyst",
    "remediation_agent": "Remediation Agent",
    "validate_fixes": "Validating fixes (SQL policy, PySpark checks, dry run)",
    "human_approval": "Waiting for human approval",
    "execute_fixes": "Executing approved fixes",
    "verify": "Verifying remediated data",
    "report": "Assembling report",
}
STATUS_BADGES = {"awaiting_approval": "🟠 Awaiting approval", "completed": "✅ Completed",
                 "incomplete": "⚠️ Incomplete (retryable)", "failed": "❌ Failed", "not_started": "⏳ Not started"}


@st.cache_resource(show_spinner=False)
def _service(data_dir: str, key_fingerprint: int) -> InvestigationService:
    """One service per data directory / key configuration (cached across reruns)."""
    return InvestigationService(Services.create())


def get_service() -> InvestigationService | None:
    """The investigation service, or ``None`` if Groq is not configured."""
    settings = get_settings()
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    if not key.strip():
        return None
    try:
        return _service(str(settings.data_dir.resolve()), hash(key))
    except LLMConfigurationError:
        return None


def _rehydrate(service: InvestigationService, run_id: str, text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return service.workspace(run_id).guard.rehydrate(text)
    except DataSentinelError:
        return text


# ---------------------------------------------------------------------- entry point


def render_investigation(file: tuple[bytes, str] | None, baseline_text: str | None, iqr: float, null_tol: float) -> None:
    """Render the AI Investigation tab."""
    service = get_service()
    if service is None:
        st.warning("AI investigation needs a Groq API key. Set `GROQ_API_KEY` in your environment or `.env` file, "
                   "then restart the app.")
        st.caption("The deterministic analysis in the other tabs works without any API key.")
        return

    notice = st.session_state.pop("inv_notice", None)
    if notice:
        st.error(notice)

    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_start(service, file, baseline_text, iqr, null_tol)
    with right:
        _render_run_picker(service)

    run_id = st.session_state.get("active_run")
    if run_id:
        st.divider()
        try:
            _render_run(service, run_id)
        except RunStoreError as exc:
            st.error(f"Run could not be loaded: {exc}")


def _render_start(service: InvestigationService, file, baseline_text, iqr, null_tol) -> None:
    st.subheader("Start an investigation")
    if file is None:
        st.info("Load a dataset in the sidebar to investigate it.")
        return
    st.caption(f"Dataset: **{file[1]}**" + (" · with schema baseline" if baseline_text else ""))
    mode = st.radio("Privacy mode for data sent to Groq", ["metadata_only", "masked_samples"], key="inv_privacy",
                    format_func=lambda m: {"metadata_only": "Metadata only (default, recommended)",
                                           "masked_samples": "Include masked sample values"}[m])
    consent = True
    if mode == "masked_samples":
        consent = st.checkbox("I consent to sending masked sample values (PII patterns masked, truncated) to Groq.",
                              key="inv_consent")
    else:
        st.caption("Only schema, aggregates, counts, value shapes and tokenised labels are sent. No raw values.")
    if st.button("🔎 Run AI investigation", type="primary", disabled=not consent, key="inv_start"):
        try:
            manifest = service.create_run(file[0], file[1], baseline_text=baseline_text, privacy_mode=mode,
                                          iqr_multiplier=iqr, max_null_pct=null_tol)
        except DataSentinelError as exc:
            st.error(f"Could not create the run: {exc}")
            return
        st.session_state["active_run"] = manifest.run_id
        _stream_with_progress(lambda cb: service.start(manifest.run_id, on_update=cb))
        st.rerun()


def _stream_with_progress(action) -> RunSnapshot | None:
    with st.status("Investigation running…", expanded=True) as status:
        def on_update(chunk: dict[str, Any]) -> None:
            for node in chunk:
                if node != "__interrupt__":
                    status.write(f"✓ {NODE_LABELS.get(node, node)}")
        try:
            snapshot = action(on_update)
        except ValueError as exc:
            # Stale action, e.g. decisions already submitted from another tab/session. Shown after the rerun.
            st.session_state["inv_notice"] = f"Action not applied: {exc} The run view below shows its current state."
            status.update(label="Action not applied", state="error")
            return None
        status.update(label=STATUS_BADGES.get(snapshot.status, snapshot.status),
                      state="error" if snapshot.error else "complete")
    return snapshot


def _render_run_picker(service: InvestigationService) -> None:
    st.subheader("Previous runs")
    runs = service.list_runs()
    if not runs:
        st.caption("No runs yet. Runs are persisted locally and survive restarts.")
        return
    labels = {m.run_id: f"{m.created_at} · {m.filename} · {m.privacy_mode}" for m in runs}
    choice = st.selectbox("Open a run", ["—", *labels], format_func=lambda r: labels.get(r, r), key="inv_pick")
    if choice != "—" and st.button("Open run", key="inv_open"):
        st.session_state["active_run"] = choice
        st.rerun()


# ---------------------------------------------------------------------- run view


def _render_run(service: InvestigationService, run_id: str) -> None:
    snapshot = service.snapshot(run_id)
    values = snapshot.values
    manifest = service.services.store.load_manifest(run_id)
    st.markdown(f"### Run `{run_id[:12]}…` — {STATUS_BADGES.get(snapshot.status, snapshot.status)}")
    st.caption(f"{manifest.filename} · privacy: {manifest.privacy_mode} · created {manifest.created_at}")

    if snapshot.status == "incomplete":
        st.warning("This run stopped before finishing (for example a Groq outage). It can continue from its last checkpoint.")
        if st.button("Retry from checkpoint", key="inv_retry"):
            _stream_with_progress(lambda cb: service.retry(run_id, on_update=cb))
            st.rerun()
    if snapshot.status == "not_started":
        return
    final = values.get("final_report")
    if final is not None and final.incomplete_agents:
        st.warning(f"Partial result: {', '.join(final.incomplete_agents)} failed or degraded. "
                   "See 'Audit & diagnostics' for details.")

    digest = values.get("digest")
    if digest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Engine quality score", "—" if digest.overall_score is None else f"{digest.overall_score:.1f}", border=True)
        c2.metric("Failed checks", len(digest.failed_checks), border=True)
        c3.metric("Evidence records", len(values.get("evidence") or []), border=True)
        c4.metric("Grounding notes", sum(1 for e in values.get("errors") or [] if e.kind == "grounding"), border=True)

    tabs = st.tabs(["Findings", "Root cause", "Schema drift", "Remediation", "Evidence", "Audit & diagnostics"])
    with tabs[0]:
        _render_findings(service, run_id, values)
    with tabs[1]:
        _render_root_cause(service, run_id, values)
    with tabs[2]:
        _render_schema(service, run_id, values)
    with tabs[3]:
        _render_remediation(service, run_id, snapshot)
    with tabs[4]:
        _render_evidence(service, run_id, values)
    with tabs[5]:
        _render_audit(service, run_id, values)


def _render_findings(service, run_id, values) -> None:
    report = values.get("investigator_report")
    if report is None:
        st.info("No investigator findings (no failed checks, or the agent did not finish).")
        return
    st.markdown(_rehydrate(service, run_id, report.summary))
    for finding in report.findings:
        badge = "🟢 confirmed" if finding.status == "confirmed" else "🟡 hypothesis"
        with st.expander(f"{finding.finding_id} · {badge} · [{finding.severity}] {_rehydrate(service, run_id, finding.title)}"):
            st.markdown(_rehydrate(service, run_id, finding.description))
            if finding.pattern:
                st.markdown(f"**Pattern:** {_rehydrate(service, run_id, finding.pattern)}")
            st.markdown(f"**Evidence:** {', '.join(f'`{e}`' for e in finding.evidence_ids) or '—'}")
            cols = st.columns(3)
            cols[0].metric("Affected rows", finding.affected_rows if finding.affected_rows is not None else "—")
            cols[1].metric("Affected %", f"{finding.affected_pct:.2f}" if finding.affected_pct is not None else "—")
            cols[2].metric("LLM estimate (uncalibrated)", f"{finding.confidence_estimate:.2f}")
            if finding.limitations:
                st.caption("Limitations: " + "; ".join(finding.limitations))
            for note in finding.grounding_notes:
                st.warning(f"Grounding: {note}")


def _render_root_cause(service, run_id, values) -> None:
    report = values.get("root_cause_report")
    if report is None:
        st.info("No root-cause analysis yet.")
        return
    st.markdown(_rehydrate(service, run_id, report.summary))
    st.markdown("#### Confirmed facts")
    for fact in report.confirmed_facts:
        st.markdown(f"- 🟢 {_rehydrate(service, run_id, fact.statement)} — evidence: "
                    f"{', '.join(f'`{e}`' for e in fact.supporting_evidence_ids)}")
    if not report.confirmed_facts:
        st.caption("None.")
    st.markdown("#### Likely causes (hypotheses)")
    for cause in report.root_causes:
        with st.expander(f"{cause.cause_id} · {cause.category} · findings {', '.join(cause.related_finding_ids) or '—'}"):
            st.markdown(_rehydrate(service, run_id, cause.explanation))
            for h in cause.hypotheses:
                st.markdown(f"- **{h.status}** · {_rehydrate(service, run_id, h.statement)} "
                            f"(LLM estimate {h.confidence_estimate:.2f}; supporting {h.supporting_evidence_ids}; "
                            f"contradicting {h.contradicting_evidence_ids})")
                if h.how_to_verify:
                    st.caption(f"How to verify: {h.how_to_verify}")
    if report.unverified_claims:
        st.markdown("#### Claims downgraded by grounding")
        for claim in report.unverified_claims:
            st.markdown(f"- 🟡 {_rehydrate(service, run_id, claim.statement)} — {'; '.join(claim.grounding_notes)}")
    impact = report.impact
    st.markdown(f"#### Impact · blast radius **{impact.blast_radius}**")
    st.markdown("\n".join(f"- {_rehydrate(service, run_id, r)}" for r in impact.downstream_risks) or "—")
    if report.limitations:
        st.caption("Limitations: " + "; ".join(report.limitations))


def _render_schema(service, run_id, values) -> None:
    report = values.get("schema_report")
    if report is None:
        st.info("No schema drift analysis (no baseline provided, or no drift detected).")
        return
    st.markdown(_rehydrate(service, run_id, report.summary))
    st.dataframe(pd.DataFrame([{
        "column": a.column, "change": a.change_type, "compatibility (deterministic)": a.compatibility,
        "status": a.status, "impact": _rehydrate(service, run_id, a.impact), "recommendation": a.recommendation,
        "evidence": ", ".join(a.evidence_ids), "grounding": "; ".join(a.grounding_notes),
    } for a in report.assessments]), hide_index=True)


def _render_remediation(service, run_id, snapshot: RunSnapshot) -> None:
    values = snapshot.values
    plan = values.get("remediation_plan")
    if plan is None:
        st.info("No remediation proposed (no confirmed, remediable findings).")
        return
    validations = {v.fix_id: v for v in values.get("fix_validations") or []}
    if snapshot.status == "awaiting_approval":
        _render_approval_form(service, run_id, snapshot)
    else:
        for fix in plan.fixes:
            v = validations.get(fix.fix_id)
            with st.expander(f"{fix.fix_id} · {fix.title} · {'✅ valid' if v and v.approvable else '⛔ blocked'}"):
                _fix_details(fix, v)
    execution = values.get("execution")
    if execution:
        st.markdown("#### Execution")
        st.dataframe(pd.DataFrame([r.model_dump() for r in execution.records]), hide_index=True)
        if execution.artifact_name:
            df = service.services.store.load_artifact(run_id, execution.artifact_name)
            st.download_button("⬇️ Download remediated dataset (CSV)", df.to_csv(index=False),
                               file_name=f"{execution.artifact_name}_{run_id[:8]}.csv", mime="text/csv")
            st.caption("The original dataset is unchanged; this is a new, separate artefact.")
    verification = values.get("verification")
    if verification:
        st.markdown("#### Verification (deterministic engine)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality score", f"{verification.score_after}",
                  delta=None if verification.score_before is None or verification.score_after is None
                  else round(verification.score_after - verification.score_before, 2), border=True)
        c2.metric("Failed checks", verification.failed_checks_after,
                  delta=verification.failed_checks_after - verification.failed_checks_before, delta_color="inverse",
                  border=True)
        c3.metric("Grade", f"{verification.grade_before} → {verification.grade_after}", border=True)
        if verification.resolved_checks:
            st.markdown("Resolved: " + ", ".join(f"`{c}`" for c in verification.resolved_checks))
        if verification.new_failures:
            st.warning("New failures: " + ", ".join(verification.new_failures))


def _fix_details(fix, validation) -> None:
    st.markdown(f"**Strategy:** {fix.strategy} · **Risk:** {fix.risk_level} · **Reversible:** {fix.reversible} · "
                f"**Addresses:** {', '.join(fix.addresses_finding_ids)}")
    st.markdown(fix.explanation)
    if fix.assumptions:
        st.caption("Assumptions: " + "; ".join(fix.assumptions))
    st.markdown("**SQL (DuckDB)**")
    st.code(fix.sql, language="sql")
    st.markdown("**PySpark (statically validated, never executed)**")
    st.code(fix.pyspark, language="python")
    if validation:
        st.caption(f"SQL SHA-256: `{validation.sql_sha256}`")
        st.dataframe(pd.DataFrame([{
            "sql ok": validation.sql_ok, "pyspark ok": validation.pyspark_ok, "dry run ok": validation.dry_run_ok,
            "columns preserved": validation.columns_preserved, "rows before": validation.rows_before,
            "rows after": validation.rows_after, "score before": validation.score_before,
            "score after": validation.score_after,
        }]), hide_index=True)
        for violation in validation.sql_violations + validation.pyspark_violations + (
                [validation.dry_run_error] if validation.dry_run_error else []):
            st.error(violation)


def _render_approval_form(service, run_id, snapshot: RunSnapshot) -> None:
    payload = snapshot.pending_approval or {}
    fixes = payload.get("fixes", [])
    st.warning("Review each fix. Nothing runs until you approve it, and approval is bound to the exact SQL shown.")
    plan = snapshot.values["remediation_plan"]
    by_id = {f.fix_id: f for f in plan.fixes}
    validations = {v.fix_id: v for v in snapshot.values.get("fix_validations") or []}
    blocked = [f for f in plan.fixes if f.fix_id not in {x["fix_id"] for x in fixes}]
    with st.form(key=f"approval_{run_id}"):
        reviewer = st.text_input("Reviewer name (required, recorded in the audit trail)", key="inv_reviewer")
        decisions = []
        for item in fixes:
            fix = by_id[item["fix_id"]]
            st.markdown(f"#### {fix.fix_id} · {fix.title}")
            _fix_details(fix, validations.get(fix.fix_id))
            choice = st.radio("Decision", ["reject", "approve"], horizontal=True, key=f"inv_decision_{fix.fix_id}")
            comment = st.text_input("Comment (optional)", key=f"inv_comment_{fix.fix_id}")
            decisions.append({"fix_id": fix.fix_id, "sql_sha256": item["sql_sha256"],
                              "decision": "approved" if choice == "approve" else "rejected", "comment": comment or None})
        submitted = st.form_submit_button("Submit decisions", type="primary")
    if blocked:
        with st.expander(f"{len(blocked)} fix(es) blocked by validation (not offered for approval)"):
            for fix in blocked:
                _fix_details(fix, validations.get(fix.fix_id))
    if submitted:
        if not reviewer.strip():
            st.error("Enter your name before submitting decisions.")
            return
        for decision in decisions:
            decision["reviewer"] = reviewer.strip()
        _stream_with_progress(lambda cb: service.resume(run_id, decisions, on_update=cb))
        st.rerun()


def _render_evidence(service, run_id, values) -> None:
    evidence = values.get("evidence") or []
    if not evidence:
        st.info("No evidence recorded.")
        return
    st.dataframe(pd.DataFrame([{"id": e.evidence_id, "source": e.source, "tool": e.tool, "by": e.produced_by,
                                "summary": _rehydrate(service, run_id, e.summary)} for e in evidence]), hide_index=True)
    chosen = st.selectbox("Inspect evidence", [e.evidence_id for e in evidence], key="inv_evidence")
    record = next(e for e in evidence if e.evidence_id == chosen)
    st.json(json.loads(json.dumps(record.model_dump(), default=str)), expanded=False)
    st.caption("This is exactly what the agents saw (privacy-filtered). Tokens such as `<column#3>` are shown unresolved here.")


def _render_audit(service, run_id, values) -> None:
    entries = service.services.audit(run_id).read()
    st.markdown("#### Audit trail (append-only)")
    st.dataframe(pd.DataFrame(entries) if entries else pd.DataFrame(columns=["timestamp", "event"]), hide_index=True)
    st.markdown("#### Agent status")
    st.dataframe(pd.DataFrame([{"agent": k, **v.model_dump()} for k, v in (values.get("agent_status") or {}).items()]),
                 hide_index=True)
    errors = values.get("errors") or []
    if errors:
        st.markdown("#### Grounding and error log")
        st.dataframe(pd.DataFrame([e.model_dump() for e in errors]), hide_index=True)
    usage = values.get("usage") or {}
    if usage:
        st.markdown("#### LLM usage")
        st.dataframe(pd.DataFrame([{"agent": k, **v.model_dump()} for k, v in usage.items()]), hide_index=True)
    st.caption(f"Model: {get_agent_settings().model or get_settings().groq_model}")
