"""Opt-in live test against the real Groq API.

Run with:  GROQ_API_KEY=... python -m pytest -m live tests/agents/test_live_groq.py -v
Excluded from the default run (see ``addopts`` in pyproject.toml).
"""

from __future__ import annotations

import os

import pytest

from datasentinel.config import Settings
from datasentinel.remediation import validate_sql
from datasentinel_agents.config import AgentSettings
from datasentinel_agents.service import InvestigationService
from datasentinel_agents.services import Services

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


def test_live_investigation_on_sample(tmp_path, sample_bytes) -> None:
    services = Services.create(settings=Settings(data_dir=tmp_path), agent_settings=AgentSettings())
    service = InvestigationService(services)
    try:
        run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
        snap = service.start(run_id)
        assert snap.status in {"awaiting_approval", "completed"}, snap.error
        ledger = {e.evidence_id for e in snap.values["evidence"]}
        report = snap.values.get("investigator_report")
        assert report is not None and report.findings
        for finding in report.findings:
            if finding.status == "confirmed":
                assert finding.evidence_ids and set(finding.evidence_ids) <= ledger
        assert any(f.issue_type == "duplicate" for f in report.findings)
        if snap.status == "awaiting_approval":
            for fix in snap.pending_approval["fixes"]:
                assert validate_sql(fix["sql"]).ok
    finally:
        service.close()
