"""End-to-end LangGraph tests with scripted LLMs and real SQLite checkpoints."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from datasentinel.schema import SchemaInferrer
from datasentinel_agents.llm import TransientLLMError
from tests.agents import scenarios as sc
from tests.agents.fakes import submit, submit_with, task_of, tool_call


def _clean_csv(extra_column: bool = False) -> bytes:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(200)],
        "amount": np.round(rng.uniform(40, 60, 200), 2),
        "region": np.resize(["N", "S", "E", "W"], 200),
    })
    if extra_column:
        df["channel"] = np.resize(["web", "store"], 200)
    return df.to_csv(index=False).encode()


def _baseline_without_extra() -> str:
    df = pd.read_csv(pd.io.common.BytesIO(_clean_csv()))
    return json.dumps(SchemaInferrer().infer(df).to_dict())


def _approve(snapshot, *fix_ids, reviewer="alice", tamper=False):
    decisions = []
    for fix in snapshot.pending_approval["fixes"]:
        if fix["fix_id"] in fix_ids:
            digest = "0" * 64 if tamper else fix["sql_sha256"]
            decisions.append({"fix_id": fix["fix_id"], "sql_sha256": digest, "decision": "approved", "reviewer": reviewer})
    return decisions


def _happy_scripts():
    return {"data_investigator": sc.investigator_script(), "root_cause_analyst": sc.root_cause_script(),
            "remediation_agent": sc.remediation_script()}


# ---------------------------------------------------------------------- full workflow


def test_full_workflow_with_restart_between_review_and_execution(make_service, sample_bytes, settings) -> None:
    service, fake = make_service(_happy_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    nodes: list[str] = []
    snap = service.start(run_id, on_update=lambda chunk: nodes.extend(chunk))
    assert snap.status == "awaiting_approval"
    assert nodes[:6] == ["prepare_context", "data_investigator", "grounding_check", "root_cause_analyst",
                         "remediation_agent", "validate_fixes"]
    assert [f["fix_id"] for f in snap.pending_approval["fixes"]] == ["FIX-1"]  # malicious FIX-2 and ungrounded FIX-3 excluded
    service.close()

    # Simulate an application restart: new service, new workspace cache, new SQLite connection, no LLM scripts.
    restarted, fake2 = make_service({})
    assert restarted.snapshot(run_id).status == "awaiting_approval"
    final = restarted.resume(run_id, _approve(snap, "FIX-1"))
    assert final.status == "completed" and fake2.calls == []
    report = final.values["final_report"]
    assert report.outcome == "remediated"
    assert report.verification.score_after > report.verification.score_before
    assert "uniqueness.duplicate_records:(dataset)" in report.verification.resolved_checks
    assert report.quality_score == restarted.workspace(run_id).result.quality.overall_score  # from the engine

    store = restarted.services.store
    assert hashlib.sha256(store.source_bytes(run_id)).hexdigest() == store.load_manifest(run_id).source_sha256
    remediated = store.load_artifact(run_id, "remediated")
    assert len(remediated) == 500 and len(restarted.workspace(run_id).result.raw_dataframe) == 506
    entries = restarted.services.audit(run_id).read()
    assert [e["event"] for e in entries] == ["approval_decision", "execution_started", "fix_execution", "artifact_saved"]
    assert entries[0]["fix_id"] == entries[2]["fix_id"] == "FIX-1"
    assert entries[1]["fix_ids"] == ["FIX-1"] and entries[3]["artifact"] == "remediated"


def test_tampered_hash_is_rejected_and_nothing_executes(make_service, sample_bytes) -> None:
    service, _ = make_service(_happy_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    final = service.resume(run_id, _approve(snap, "FIX-1", tamper=True))
    report = final.values["final_report"]
    assert report.outcome == "fixes_rejected" and report.execution is None
    assert report.approvals[0].decision == "rejected" and "hash mismatch" in report.approvals[0].system_note
    assert service.services.store.load_manifest(run_id).artifacts == []


def test_sql_swapped_after_review_is_blocked_at_both_gates(make_service, sample_bytes) -> None:
    from datasentinel_agents.nodes.control import make_execute_node, parse_decisions

    service, _ = make_service(_happy_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    state = dict(service.snapshot(run_id).values)
    reviewed_hash = snap.pending_approval["fixes"][0]["sql_sha256"]
    plan = state["remediation_plan"].model_copy(deep=True)
    plan.fixes[0].sql = "SELECT * FROM dataset WHERE order_id < 100100"  # swapped after the reviewer saw it
    state["remediation_plan"] = plan

    # Gate 1: decisions are re-checked against the current SQL.
    decisions = parse_decisions({"decisions": _approve(snap, "FIX-1")}, state)
    assert decisions[0].decision == "rejected" and "hash mismatch" in decisions[0].system_note

    # Gate 2: even with a forged approval record, the executor re-hashes the SQL it would run.
    forged = decisions[0].model_copy(update={"decision": "approved", "sql_sha256": reviewed_hash})
    result = make_execute_node(service.services)({**state, "approvals": [forged]})["execution"]
    assert result.records[0].status == "blocked" and "hash" in result.records[0].reason
    assert result.artifact_name is None


@pytest.mark.parametrize("decisions, note", [
    ([], "No decision provided"),
    ([{"fix_id": "FIX-1", "decision": "approved", "reviewer": "  "}], "Reviewer identity"),
])
def test_missing_decisions_or_reviewer_reject(make_service, sample_bytes, decisions, note) -> None:
    service, _ = make_service(_happy_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    if decisions:
        decisions[0]["sql_sha256"] = snap.pending_approval["fixes"][0]["sql_sha256"]
    final = service.resume(run_id, decisions)
    assert note in final.values["final_report"].approvals[0].system_note
    assert final.values["final_report"].outcome == "fixes_rejected"


def test_explicit_rejection(make_service, sample_bytes) -> None:
    service, _ = make_service(_happy_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    fix = snap.pending_approval["fixes"][0]
    final = service.resume(run_id, [{"fix_id": "FIX-1", "sql_sha256": fix["sql_sha256"], "decision": "rejected",
                                     "reviewer": "bob", "comment": "Need upstream fix instead"}])
    approval = final.values["final_report"].approvals[0]
    assert approval.decision == "rejected" and approval.comment == "Need upstream fix instead"
    with pytest.raises(ValueError):
        service.resume(run_id, [])


# ---------------------------------------------------------------------- routing


def test_clean_dataset_skips_all_agents(make_service) -> None:
    service, fake = make_service({})
    run_id = service.create_run(_clean_csv(), "clean.csv").run_id
    snap = service.start(run_id)
    assert snap.status == "completed" and fake.calls == []
    assert snap.values["final_report"].outcome == "no_issues"
    with pytest.raises(ValueError):
        service.start(run_id)


def test_drift_only_runs_schema_analyst_and_grounds_it(make_service) -> None:
    rca = [submit({"summary": "Non-breaking addition.", "impact": {"blast_radius": "low"}, "confidence_estimate": 0.8})]
    service, fake = make_service({"schema_analyst": sc.schema_script(), "root_cause_analyst": rca})
    run_id = service.create_run(_clean_csv(extra_column=True), "clean.csv", baseline_text=_baseline_without_extra()).run_id
    snap = service.start(run_id)
    assert snap.status == "completed" and set(fake.calls) == {"schema_analyst", "root_cause_analyst"}
    report = snap.values["schema_report"]
    assert report.drift_detected is True  # overwritten from the drift engine
    real = next(a for a in report.assessments if a.column == "channel")
    assert real.compatibility == "non_breaking" and real.status == "confirmed"
    ghost = next(a for a in report.assessments if a.column == "imaginary_column")
    assert ghost.status == "hypothesis" and ghost.evidence_ids == []
    assert snap.values["final_report"].outcome == "investigated"


def test_failures_and_drift_run_agents_in_parallel(make_service, sample_bytes) -> None:
    baseline = json.dumps({"columns": {"order_id": {"type": "numeric", "unique": True}}, "allow_extra_columns": True})
    scripts = {**_happy_scripts(), "schema_analyst": sc.schema_script()}
    service, fake = make_service(scripts)
    run_id = service.create_run(sample_bytes, "customer_orders.csv", baseline_text=baseline).run_id
    nodes: list[str] = []
    snap = service.start(run_id, on_update=lambda chunk: nodes.extend(chunk))
    # Fan-out: both agents ran before grounding, and grounding ran once (i.e. both were in the same superstep).
    assert nodes.count("grounding_check") == 1
    assert max(nodes.index("data_investigator"), nodes.index("schema_analyst")) < nodes.index("grounding_check")
    statuses = snap.values["agent_status"]
    assert statuses["data_investigator"].status == statuses["schema_analyst"].status == "succeeded"
    rca_task = task_of(fake.models["root_cause_analyst"].received[0])
    assert rca_task["schema_analysis"]["drift_detected"] is True


def test_followup_round_is_bounded(make_service, sample_bytes) -> None:
    scripts = {
        "data_investigator": sc.investigator_script() + sc.followup_investigator_script(),
        "root_cause_analyst": sc.root_cause_script(followup=True) + sc.root_cause_script(followup=True),
        "remediation_agent": sc.remediation_script(),
    }
    service, fake = make_service(scripts)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert fake.calls.count("data_investigator") == 2 and fake.calls.count("root_cause_analyst") == 2
    assert snap.values["followup_rounds"] == 1 and snap.status == "awaiting_approval"
    findings = snap.values["investigator_report"].findings
    assert [f.finding_id for f in findings] == ["F1", "F2", "F3", "F1b"]
    followup_task = task_of(fake.models["data_investigator"].received[-1])
    assert followup_task["followup_requests"][0]["column"] == "quantity"


def test_remediation_repair_round_then_report(make_service, sample_bytes) -> None:
    def bad_plan(messages):
        return {"fixes": [{"fix_id": "FIX-1", "addresses_finding_ids": ["F1"], "title": "drop column",
                           "strategy": "filter", "sql": "SELECT order_id FROM dataset", "pyspark": "result = df",
                           "explanation": "x", "risk_level": "low", "reversible": True, "expected_effect": "x"}]}

    def check_feedback(messages):
        violations = task_of(messages)["previous_attempt_violations"]
        assert violations and violations[0]["columns_preserved"] is False
        return bad_plan(messages)

    scripts = {"data_investigator": sc.investigator_script(), "root_cause_analyst": sc.root_cause_script(),
               "remediation_agent": [submit_with(bad_plan), submit_with(check_feedback)]}
    service, fake = make_service(scripts)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert fake.calls.count("remediation_agent") == 2
    assert snap.status == "completed" and snap.values["final_report"].outcome == "no_valid_fixes"


# ---------------------------------------------------------------------- failures and retries


def test_transient_error_is_retried(make_service, sample_bytes) -> None:
    scripts = _happy_scripts()
    scripts["data_investigator"] = [TransientLLMError("429 rate limited"), *sc.investigator_script()]
    service, _ = make_service(scripts)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert snap.status == "awaiting_approval"
    assert snap.values["agent_status"]["data_investigator"].status == "succeeded"


def test_exhausted_retries_leave_a_resumable_run(make_service, sample_bytes) -> None:
    scripts = _happy_scripts()
    scripts["data_investigator"] = [TransientLLMError("503")] * 2
    service, fake = make_service(scripts, llm_retry_attempts=2)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert snap.status == "incomplete" and "TransientLLMError" in snap.error
    assert snap.next_nodes == ("data_investigator",)
    fake.models["data_investigator"].responses.extend(sc.investigator_script())  # provider recovered
    assert service.retry(run_id).status == "awaiting_approval"


def test_permanent_agent_failure_degrades_gracefully(make_service, sample_bytes) -> None:
    rca = [submit({"summary": "No findings available.", "impact": {"blast_radius": "low"}, "confidence_estimate": 0.2,
                   "limitations": ["Investigator failed"]})]
    service, _ = make_service({"data_investigator": [ValueError("malformed provider response")], "root_cause_analyst": rca})
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert snap.status == "completed"
    assert snap.values["agent_status"]["data_investigator"].status == "failed"
    assert any(e.kind == "internal" and e.agent == "data_investigator" for e in snap.values["errors"])
    assert snap.values["final_report"].outcome == "investigated"


def test_invalid_run_ids_rejected(make_service) -> None:
    service, _ = make_service({})
    with pytest.raises(Exception):
        service.start("../../etc/passwd")
