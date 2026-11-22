"""Offline end-to-end workflow audit (scripted LLMs, real SQLite checkpoints, real engine).

Covers the full node/transition map, a complete four-agent investigation, grounding of
fabricated claims, remediation gates, rejection, interrupt/resume, per-stage failures,
source immutability, and audit/run-history consistency across failures and restarts.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from datasentinel.config import Settings
from datasentinel.pipeline import analyze_dataset
from datasentinel.remediation import SandboxError
from datasentinel_agents.config import AgentSettings
from datasentinel_agents.graph import build_graph
from datasentinel_agents.llm import TransientLLMError, groq_llm_factory
from datasentinel_agents.nodes.common import analyze_frame
from datasentinel_agents.services import Services
from tests.agents import scenarios as sc
from tests.agents.fakes import submit
from tests.conftest import LIVE_CALLS_DISABLED

EXPECTED_EDGES = {
    ("__start__", "prepare_context", False),
    ("prepare_context", "data_investigator", True), ("prepare_context", "schema_analyst", True),
    ("prepare_context", "report", True),
    ("data_investigator", "grounding_check", False), ("schema_analyst", "grounding_check", False),
    ("grounding_check", "root_cause_analyst", False),
    ("root_cause_analyst", "data_investigator", True), ("root_cause_analyst", "remediation_agent", True),
    ("root_cause_analyst", "report", True),
    ("remediation_agent", "validate_fixes", False),
    ("validate_fixes", "human_approval", True), ("validate_fixes", "remediation_agent", True),
    ("validate_fixes", "report", True),
    ("human_approval", "execute_fixes", True), ("human_approval", "report", True),
    ("execute_fixes", "verify", False), ("verify", "report", False), ("report", "__end__", False),
}
RCA_MINIMAL = [submit({"summary": "Limited analysis.", "impact": {"blast_radius": "low"}, "confidence_estimate": 0.2})]


# ---------------------------------------------------------------------- helpers


def _source_fingerprint(service, run_id: str) -> tuple[str, int, int]:
    manifest = service.services.store.load_manifest(run_id)
    path = service.services.store.run_dir(run_id) / f"source{Path(manifest.filename).suffix}"
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), path.stat().st_mtime_ns, len(data)


def _approve_all(snapshot, reviewer: str = "alice") -> list[dict]:
    return [{"fix_id": f["fix_id"], "sql_sha256": f["sql_sha256"], "decision": "approved", "reviewer": reviewer}
            for f in snapshot.pending_approval["fixes"]]


def _events(service, run_id: str) -> list[str]:
    return [e["event"] for e in service.services.audit(run_id).read()]


def _assert_artifacts_consistent(service, run_id: str) -> None:
    """Every artefact listed in the manifest exists on disk (unlisted orphans are tolerated, never overwritten)."""
    store = service.services.store
    on_disk = {p.stem for p in (store.run_dir(run_id) / "artifacts").glob("*.parquet")}
    assert set(store.load_manifest(run_id).artifacts) <= on_disk


# ---------------------------------------------------------------------- 1. workflow structure


def test_graph_transitions_match_documented_workflow(settings, agent_settings) -> None:
    services = Services.create(llm_factory=lambda agent: None, settings=settings, agent_settings=agent_settings)
    graph = build_graph(services).get_graph()
    assert {(e.source, e.target, bool(e.conditional)) for e in graph.edges} == EXPECTED_EDGES
    assert set(graph.nodes) == {n for edge in EXPECTED_EDGES for n in edge[:2]}


def test_offline_suite_cannot_reach_groq() -> None:
    model = groq_llm_factory(Settings(_env_file=None, GROQ_API_KEY="gsk_dummy"), AgentSettings(_env_file=None))("x")
    with pytest.raises(RuntimeError, match="disabled in the offline test suite"):
        model.invoke([HumanMessage(content="hello")])
    assert LIVE_CALLS_DISABLED


# ---------------------------------------------------------------------- 2-5. complete successful investigation


def test_complete_four_agent_investigation_through_verification(make_service, sample_bytes) -> None:
    service, fake = make_service(sc.full_scripts(with_schema=True))
    run_id = service.create_run(sample_bytes, "customer_orders.csv", baseline_text=sc.SAMPLE_BASELINE).run_id
    before = _source_fingerprint(service, run_id)
    nodes: list[str] = []

    snap = service.start(run_id, on_update=lambda chunk: nodes.extend(chunk))
    assert snap.status == "awaiting_approval" and snap.next_nodes == ("human_approval",)
    assert nodes[0] == "prepare_context" and {"data_investigator", "schema_analyst"} <= set(nodes)
    assert nodes.count("grounding_check") == 1 and nodes[-1] == "__interrupt__"
    assert _events(service, run_id) == []  # nothing audited, nothing executed before approval

    values = snap.values
    ledger = {e.evidence_id for e in values["evidence"]}
    findings = {f.finding_id: f for f in values["investigator_report"].findings}
    # Fabricated evidence, unsupported claims and invented numbers were grounded.
    assert findings["F3"].status == "hypothesis" and findings["F3"].affected_rows is None
    assert all(set(f.evidence_ids) <= ledger for f in findings.values())
    rca = values["root_cause_report"]
    assert [c.statement for c in rca.unverified_claims] == ["The CRM double-exported orders."]
    assert rca.impact.affected_rows_pct is None and rca.root_causes[0].related_finding_ids == ["F1"]
    schema = values["schema_report"]
    assert schema.drift_detected is True
    confirmed = [a for a in schema.assessments if a.status == "confirmed"]
    assert {(a.change_type, a.column) for a in confirmed if a.compatibility == "breaking"} == {
        ("removed", "legacy_code"), ("uniqueness_changed", "order_id")}
    # Columns absent from the (partial) baseline are additions; the deterministic class overrides the LLM's.
    assert {a.change_type for a in confirmed if a.compatibility == "non_breaking"} == {"added"}
    assert all(a.status == "hypothesis" for a in schema.assessments if a.column == "imaginary_column")
    # Remediation gates: only the safe, grounded fix is offered.
    validations = {v.fix_id: v for v in values["fix_validations"]}
    assert validations["FIX-1"].approvable and not validations["FIX-2"].approvable
    assert "FIX-3" not in validations
    assert [f["fix_id"] for f in snap.pending_approval["fixes"]] == ["FIX-1"]

    final = service.resume(run_id, _approve_all(snap))
    assert final.status == "completed"
    report = final.values["final_report"]
    assert report.outcome == "remediated" and report.incomplete_agents == []
    assert {k: v.status for k, v in report.agent_status.items()} == {
        "data_investigator": "succeeded", "schema_analyst": "succeeded",
        "root_cause_analyst": "succeeded", "remediation_agent": "succeeded"}

    # Scores are engine results, recomputed independently here.
    workspace = service.workspace(run_id)
    remediated = service.services.store.load_artifact(run_id, "remediated")
    expected_after = analyze_frame(workspace, remediated, "check").quality.overall_score
    assert report.quality_score == workspace.result.quality.overall_score
    assert report.verification.score_after == expected_after
    assert len(remediated) == 500 and _source_fingerprint(service, run_id) == before
    assert _events(service, run_id) == ["approval_decision", "execution_started", "fix_execution", "artifact_saved"]
    assert "pyspark" not in sys.modules  # generated PySpark was only parsed, never imported or executed


def test_engine_severity_overrides_llm_claim(make_service, sample_bytes) -> None:
    service, _ = make_service({"data_investigator": sc.wrong_severity_investigator_script(),
                               "root_cause_analyst": RCA_MINIMAL})
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    finding = snap.values["investigator_report"].findings[0]
    assert finding.severity == "medium" and finding.status == "confirmed"
    assert any("engine severity" in note for note in finding.grounding_notes)


# ---------------------------------------------------------------------- 6. rejection and approval integrity


def test_rejection_executes_nothing(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    before = _source_fingerprint(service, run_id)
    snap = service.start(run_id)
    fix = snap.pending_approval["fixes"][0]
    final = service.resume(run_id, [{"fix_id": fix["fix_id"], "sql_sha256": fix["sql_sha256"],
                                     "decision": "rejected", "reviewer": "bob"}])
    report = final.values["final_report"]
    assert report.outcome == "fixes_rejected" and report.execution is None and report.verification is None
    assert report.agent_status["schema_analyst"].status == "skipped"  # never routed to (no baseline)
    assert report.incomplete_agents == []
    assert service.services.store.load_manifest(run_id).artifacts == []
    assert _events(service, run_id) == ["approval_decision"]
    assert _source_fingerprint(service, run_id) == before


def test_crafted_approval_of_blocked_fix_is_ignored(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    blocked = next(v for v in snap.values["fix_validations"] if v.fix_id == "FIX-2")
    final = service.resume(run_id, [{"fix_id": "FIX-2", "sql_sha256": blocked.sql_sha256, "decision": "approved",
                                     "reviewer": "mallory"}])
    report = final.values["final_report"]
    assert [(a.fix_id, a.decision) for a in report.approvals] == [("FIX-1", "rejected")]
    assert report.execution is None


# ---------------------------------------------------------------------- 7. interrupt and resume


def test_interrupt_survives_multiple_restarts(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    service.close()
    for _ in range(2):  # restart twice while waiting for a human
        restarted, fake = make_service({})
        assert restarted.snapshot(run_id).pending_approval["fixes"][0]["sql_sha256"] == \
            snap.pending_approval["fixes"][0]["sql_sha256"]
        restarted.close()
    final_service, fake = make_service({})
    assert final_service.resume(run_id, _approve_all(snap)).status == "completed" and fake.calls == []


# ---------------------------------------------------------------------- 8. failures at each stage


@pytest.mark.parametrize("agent, scripts, baseline", [
    ("data_investigator", {"data_investigator": [ValueError("bad")], "root_cause_analyst": RCA_MINIMAL}, None),
    ("schema_analyst", {**sc.full_scripts(), "schema_analyst": [ValueError("bad")]}, sc.SAMPLE_BASELINE),
    ("root_cause_analyst", {"data_investigator": sc.investigator_script(), "root_cause_analyst": [ValueError("bad")],
                            "remediation_agent": sc.remediation_script()}, None),
    ("remediation_agent", {"data_investigator": sc.investigator_script(), "root_cause_analyst": sc.root_cause_script(),
                           "remediation_agent": [ValueError("bad")]}, None),
])
def test_permanent_agent_failure_is_reported_clearly(make_service, sample_bytes, agent, scripts, baseline) -> None:
    service, _ = make_service(scripts)
    run_id = service.create_run(sample_bytes, "customer_orders.csv", baseline_text=baseline).run_id
    snap = service.start(run_id)
    if snap.status == "awaiting_approval":  # later stages still produced a valid fix: finish the run
        snap = service.resume(run_id, _approve_all(snap))
    assert snap.status == "completed"
    report = snap.values["final_report"]
    assert agent in report.incomplete_agents and report.agent_status[agent].status == "failed"
    assert any(e.agent == agent and e.kind == "internal" for e in report.errors)


@pytest.mark.parametrize("agent", ["data_investigator", "root_cause_analyst", "remediation_agent"])
def test_transient_failure_at_each_agent_is_resumable(make_service, sample_bytes, agent) -> None:
    scripts = sc.full_scripts()
    good = scripts[agent]
    scripts[agent] = [TransientLLMError("503")] * 2
    service, fake = make_service(scripts, llm_retry_attempts=2)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    assert snap.status == "incomplete" and snap.next_nodes == (agent,) and "TransientLLMError" in snap.error
    fake.models[agent].responses.extend(good)
    assert service.retry(run_id).status == "awaiting_approval"


def test_deterministic_validation_failure_is_resumable(make_service, sample_bytes, monkeypatch) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    original = service.services.sandbox.run
    monkeypatch.setattr(service.services.sandbox, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    snap = service.start(run_id)
    assert snap.status == "incomplete" and snap.next_nodes == ("validate_fixes",) and "disk full" in snap.error
    monkeypatch.setattr(service.services.sandbox, "run", original)
    assert service.retry(run_id).status == "awaiting_approval"


def test_all_approved_fixes_failing_at_execution(make_service, sample_bytes, monkeypatch) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)

    def fail(*args, **kwargs):
        raise SandboxError("Query failed: out of memory")

    monkeypatch.setattr(service.services.sandbox, "run", fail)
    report = service.resume(run_id, _approve_all(snap)).values["final_report"]
    assert report.outcome == "execution_failed" and report.verification is None
    assert [r.status for r in report.execution.records] == ["failed"]
    assert service.services.store.load_manifest(run_id).artifacts == []
    assert _events(service, run_id) == ["approval_decision", "execution_started", "fix_execution"]


def test_verify_failure_does_not_re_execute_on_retry(make_service, sample_bytes, monkeypatch) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    store = service.services.store
    original = store.load_artifact
    monkeypatch.setattr(store, "load_artifact", lambda *a, **k: (_ for _ in ()).throw(OSError("I/O error")))
    snap = service.resume(run_id, _approve_all(snap))
    assert snap.status == "incomplete" and snap.next_nodes == ("verify",)
    monkeypatch.setattr(store, "load_artifact", original)
    assert service.retry(run_id).status == "completed"
    assert _events(service, run_id).count("execution_started") == 1  # execution was checkpointed, not repeated
    assert store.load_manifest(run_id).artifacts == ["remediated"]


# ---------------------------------------------------------------------- 9-10. immutability, audit, history


def test_crash_after_artifact_write_recovers_without_overwriting(make_service, sample_bytes, monkeypatch) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    before = _source_fingerprint(service, run_id)
    snap = service.start(run_id)
    store = service.services.store

    def crash_after_write(rid, name, df):  # file written, process dies before the manifest is updated
        df.to_parquet(store.run_dir(rid) / "artifacts" / f"{name}.parquet", index=False)
        raise OSError("power loss")

    monkeypatch.setattr(store, "save_artifact", crash_after_write)
    assert service.resume(run_id, _approve_all(snap)).status == "incomplete"
    orphan = store.run_dir(run_id) / "artifacts" / "remediated.parquet"
    orphan_bytes = orphan.read_bytes()
    monkeypatch.undo()
    service.close()

    restarted, fake = make_service({})
    final = restarted.retry(run_id)
    assert final.status == "completed" and fake.calls == []
    assert final.values["final_report"].execution.artifact_name == "remediated_2"
    assert orphan.read_bytes() == orphan_bytes  # the orphan was neither reused nor overwritten
    assert restarted.services.store.load_manifest(run_id).artifacts == ["remediated_2"]
    assert _events(restarted, run_id) == ["approval_decision", "execution_started", "fix_execution",
                                         "execution_started", "fix_execution", "artifact_saved"]
    assert _source_fingerprint(restarted, run_id) == before
    _assert_artifacts_consistent(restarted, run_id)


def test_tampered_source_blocks_execution_after_restart(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    service.close()
    source = service.services.store.run_dir(run_id) / "source.csv"
    source.write_bytes(source.read_bytes().replace(b"Electronics", b"Electronicz"))

    restarted, _ = make_service({})
    after = restarted.resume(run_id, _approve_all(snap))
    assert after.status == "incomplete" and "integrity" in after.error
    assert restarted.services.store.load_manifest(run_id).artifacts == []
    assert "execution_started" not in _events(restarted, run_id)


def test_tampered_source_before_start_fails_without_llm_calls(make_service, sample_bytes) -> None:
    service, fake = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    service.services.workspaces.evict(run_id)
    (service.services.store.run_dir(run_id) / "source.csv").write_bytes(b"order_id\n1\n")
    snap = service.start(run_id)
    assert snap.status == "incomplete" and "integrity" in snap.error and fake.calls == []


@pytest.mark.parametrize("kwargs, error", [
    ({"baseline_text": "{not json"}, "SchemaContractError"),
    ({"data": b"", "filename": "empty.csv"}, "EmptyDatasetError"),
    ({"filename": "data.xlsx"}, "UnsupportedFileTypeError"),
])
def test_invalid_inputs_never_create_runs(make_service, sample_bytes, kwargs, error) -> None:
    service, _ = make_service({})
    args = {"data": sample_bytes, "filename": "customer_orders.csv", **kwargs}
    with pytest.raises(Exception) as excinfo:
        service.create_run(**args)
    assert type(excinfo.value).__name__ == error
    assert service.list_runs() == []


def test_run_history_is_consistent_across_failures_and_restarts(make_service, sample_bytes) -> None:
    service, fake = make_service({
        "data_investigator": sc.investigator_script() * 3 + [TransientLLMError("x")] * 3,
        "root_cause_analyst": sc.root_cause_script() * 3,
        "remediation_agent": sc.remediation_script() * 3,
    })
    completed = service.create_run(sample_bytes, "a.csv").run_id
    service.resume(completed, _approve_all(service.start(completed)))
    rejected = service.create_run(sample_bytes, "b.csv").run_id
    service.start(rejected)
    service.resume(rejected, [])
    waiting = service.create_run(sample_bytes, "c.csv").run_id
    service.start(waiting)
    broken = service.create_run(sample_bytes, "d.csv").run_id
    assert service.start(broken).status == "incomplete"
    service.close()

    restarted, _ = make_service({})
    statuses = {m.run_id: restarted.snapshot(m.run_id).status for m in restarted.list_runs()}
    assert statuses == {completed: "completed", rejected: "completed", waiting: "awaiting_approval",
                        broken: "incomplete"}
    assert _events(restarted, completed)[-1] == "artifact_saved"
    assert _events(restarted, rejected) == ["approval_decision"]
    assert _events(restarted, waiting) == _events(restarted, broken) == []
    for run_id in statuses:
        _assert_artifacts_consistent(restarted, run_id)
        manifest = restarted.services.store.load_manifest(run_id)
        assert hashlib.sha256(restarted.services.store.source_bytes(run_id)).hexdigest() == manifest.source_sha256
