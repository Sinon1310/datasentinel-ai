"""The investigation workflow stepper reflects only real checkpointed state (scripted LLMs, no network)."""

from __future__ import annotations

from app.components.workflow import derive_stages
from datasentinel_agents.llm import TransientLLMError
from tests.agents import scenarios as sc
from tests.agents.fakes import submit


def _by_key(snapshot) -> dict[str, str]:
    return {s.key: s.status for s in derive_stages(snapshot.values, snapshot.status, snapshot.next_nodes)}


def test_awaiting_approval_then_completed(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    stages = _by_key(snap)
    assert stages["prepare"] == stages["investigate"] == stages["grounding"] == stages["root_cause"] == "done"
    assert stages["remediation"] == stages["validation"] == "done"
    assert stages["approval"] == "waiting"
    assert stages["execution"] == stages["verification"] == stages["report"] == "not_reached"
    assert stages["schema"] == "not_reached"  # still pending in state; becomes "skipped" when the run finishes

    fix = snap.pending_approval["fixes"][0]
    final = service.resume(run_id, [{"fix_id": fix["fix_id"], "sql_sha256": fix["sql_sha256"],
                                     "decision": "approved", "reviewer": "alice"}])
    stages = _by_key(final)
    assert stages["schema"] == "skipped"
    assert all(stages[k] == "done" for k in ("approval", "execution", "verification", "report"))


def test_rejection_marks_execution_skipped(make_service, sample_bytes) -> None:
    service, _ = make_service(sc.full_scripts())
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    service.start(run_id)
    stages = _by_key(service.resume(run_id, []))
    assert stages["approval"] == "done" and stages["execution"] == stages["verification"] == "skipped"


def test_interrupted_and_failed_stages(make_service, sample_bytes) -> None:
    scripts = sc.full_scripts()
    scripts["data_investigator"] = [TransientLLMError("503")] * 2
    service, _ = make_service(scripts, llm_retry_attempts=2)
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    stages = _by_key(service.start(run_id))
    assert stages["prepare"] == "done" and stages["investigate"] == "interrupted"
    assert stages["root_cause"] == "not_reached"

    rca = [submit({"summary": "s", "impact": {"blast_radius": "low"}, "confidence_estimate": 0.1})]
    failing, _ = make_service({"data_investigator": [ValueError("boom")], "root_cause_analyst": rca})
    run_id = failing.create_run(sample_bytes, "customer_orders.csv").run_id
    stages = _by_key(failing.start(run_id))
    assert stages["investigate"] == "failed" and stages["root_cause"] == "done" and stages["report"] == "done"


def test_clean_dataset_skips_everything_after_prepare(make_service) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    clean = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(200)],
                          "amount": np.round(rng.uniform(40, 60, 200), 2),
                          "region": np.resize(["N", "S", "E", "W"], 200)}).to_csv(index=False).encode()
    service, _ = make_service({})
    run_id = service.create_run(clean, "clean.csv").run_id
    stages = _by_key(service.start(run_id))
    assert stages["prepare"] == stages["report"] == "done"
    assert {stages[k] for k in stages if k not in ("prepare", "report")} == {"skipped"}
