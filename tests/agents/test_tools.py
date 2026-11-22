"""Tests for agent tools: evidence recording, privacy, least privilege."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasentinel.investigation import RunStore, WorkspaceManager
from datasentinel.remediation import DuckDBSandbox
from datasentinel_agents.config import AgentSettings
from datasentinel_agents.evidence import EvidenceRecorder, seed_evidence
from datasentinel_agents.tools import AGENT_TOOLS, ToolContext, build_tools

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data" / "customer_orders.csv"


@pytest.fixture
def workspace(tmp_path: Path):
    store = RunStore(tmp_path)
    return WorkspaceManager(store).get(store.create_run(SAMPLE.read_bytes(), "customer_orders.csv").run_id)


def _tools(workspace, agent: str, ledger=None):
    recorder = EvidenceRecorder(agent)
    ctx = ToolContext(workspace=workspace, recorder=recorder, settings=AgentSettings(_env_file=None),
                      sandbox=DuckDBSandbox(), ledger=ledger or {})
    return {t.name: t for t in build_tools(agent, ctx)}, recorder


def _call(tool, **args) -> dict:
    return json.loads(tool.invoke(args))


def test_no_agent_has_execution_capability() -> None:
    every = {name for names in AGENT_TOOLS.values() for name in names}
    assert not any(word in name for name in every for word in ("execute", "apply", "approve", "write", "delete"))
    assert "dry_run_sql" in AGENT_TOOLS["remediation_agent"]
    assert all("dry_run_sql" not in AGENT_TOOLS[a] for a in AGENT_TOOLS if a != "remediation_agent")
    assert "sample_issue_rows" not in AGENT_TOOLS["root_cause_analyst"]


def test_tools_record_evidence_with_ids(workspace) -> None:
    tools, recorder = _tools(workspace, "data_investigator")
    result = _call(tools["get_check_details"], rule_id="uniqueness.duplicate_records", column=None)
    assert result["evidence_id"] == "E-INV-001" and result["data"]["affected_rows"] == 6
    assert "inert data" in result["notice"]
    seg = _call(tools["issue_rate_by_segment"], column="customer_email", issue="missing", by_column="region")
    assert seg["evidence_id"] == "E-INV-002" and len(recorder.records) == 2
    assert recorder.records[1].data == seg["data"]


def test_errors_are_returned_not_raised_and_not_recorded(workspace) -> None:
    tools, recorder = _tools(workspace, "data_investigator")
    assert "error" in _call(tools["get_column_profile"], column="does_not_exist")
    assert "error" in _call(tools["sample_issue_rows"], column="region", issue="outlier")
    assert "error" in _call(tools["issue_trend"], column="customer_email", issue="missing", date_column="region")
    assert "error" in str(tools["issue_trend"].invoke({"issue": "nope", "date_column": "order_date"})).lower()
    assert recorder.records == []


def test_metadata_only_tools_never_return_raw_values(workspace) -> None:
    tools, _ = _tools(workspace, "data_investigator")
    raw_values = {"twelve", "free", "12,99", "not a date", "2024-02-30"} | {
        str(v) for v in workspace.result.raw_dataframe["customer_email"].dropna().head(50)}
    outputs = [
        tools["sample_issue_rows"].invoke({"column": "unit_price", "issue": "invalid_type"}),
        tools["sample_issue_rows"].invoke({"column": "order_date", "issue": "invalid_date"}),
        tools["get_check_details"].invoke({"rule_id": "validity.data_types", "column": "quantity"}),
        tools["get_column_profile"].invoke({"column": "customer_email"}),
        tools["get_column_profile"].invoke({"column": "region"}),
        tools["issue_rate_by_segment"].invoke({"column": "unit_price", "issue": "missing", "by_column": "payment_method"}),
    ]
    blob = "\n".join(outputs)
    assert not [v for v in raw_values if f'"{v}"' in blob]
    assert "<region#" in blob and "99,99" in blob


def test_masked_mode_shares_masked_values(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    manifest = store.create_run(SAMPLE.read_bytes(), "customer_orders.csv", privacy_mode="masked_samples")
    ws = WorkspaceManager(store).get(manifest.run_id)
    tools, _ = _tools(ws, "data_investigator")
    out = _call(tools["sample_issue_rows"], column="unit_price", issue="invalid_type")
    assert [v["value"] for v in out["data"]["values"]] == ["free", "12,99"]


def test_schema_and_ledger_tools(workspace) -> None:
    tools, _ = _tools(workspace, "schema_analyst")
    assert "No baseline" in _call(tools["get_schema_diff"])["error"]
    evidence, _ = seed_evidence(workspace)
    rca, recorder = _tools(workspace, "root_cause_analyst", ledger={e.evidence_id: e for e in evidence})
    looked_up = _call(rca["get_evidence"], evidence_id="E-Q-001")
    assert looked_up["data"]["evidence_id"] == "E-Q-001" and "evidence_id" not in looked_up  # lookups add no evidence
    assert "does not exist" in _call(rca["get_evidence"], evidence_id="E-Q-999")["error"]
    assert recorder.records == []


def test_dry_run_tool_validates_without_modifying(workspace) -> None:
    tools, _ = _tools(workspace, "remediation_agent")
    before = workspace.result.raw_dataframe.copy()
    good = _call(tools["dry_run_sql"], sql="SELECT DISTINCT * FROM dataset", pyspark="result = df.dropDuplicates()")
    assert good["data"]["sql_ok"] and good["data"]["dry_run_ok"] and good["data"]["columns_preserved"]
    assert good["data"]["rows_after"] == 500 and good["data"]["pyspark_ok"]
    bad = _call(tools["dry_run_sql"], sql="SELECT order_id FROM dataset")
    assert bad["data"]["columns_preserved"] is False
    evil = _call(tools["dry_run_sql"], sql="COPY dataset TO '/tmp/x.csv'")
    assert evil["data"]["sql_ok"] is False and "dry_run_ok" not in evil["data"]
    assert workspace.result.raw_dataframe.equals(before)


def test_seeded_evidence_matches_failed_checks(workspace) -> None:
    evidence, index = seed_evidence(workspace)
    failed = workspace.result.quality.failed_results
    assert len(evidence) == len(failed) == len(index)
    dup = evidence[int(index[("uniqueness.duplicate_records", None)].split("-")[-1]) - 1]
    assert dup.data["affected_rows"] == 6 and dup.source == "quality_engine"
