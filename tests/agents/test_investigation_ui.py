"""Headless Streamlit tests for the AI Investigation tab, including human approval through the UI."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st

from datasentinel.config import get_settings
from datasentinel.remediation import AuditLog
from datasentinel_agents.config import get_agent_settings
from tests.agents import scenarios as sc

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


@pytest.fixture
def ui_env(monkeypatch, settings):
    def _apply(with_key: bool) -> None:
        monkeypatch.setenv("DATASENTINEL_DATA_DIR", str(settings.data_dir))
        # Set (never delete) both key variables: process env vars override a developer's local .env file,
        # whereas deleting them would let Settings fall back to a real key stored in .env.
        key = "gsk_dummy_key_for_ui_tests" if with_key else ""
        monkeypatch.setenv("GROQ_API_KEY", key)
        monkeypatch.setenv("DATASENTINEL_GROQ_API_KEY", key)
        get_settings.cache_clear()
        get_agent_settings.cache_clear()
        st.cache_resource.clear()

    yield _apply
    get_settings.cache_clear()
    get_agent_settings.cache_clear()
    st.cache_resource.clear()


def test_tab_explains_missing_api_key(ui_env) -> None:
    ui_env(with_key=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="sample_choice").select("Customer orders (CSV)").run()
    assert not at.exception
    assert any("GROQ_API_KEY" in w.value for w in at.warning)
    # No service is built and no investigation can be started, so no Groq call is possible.
    assert not [b for b in at.button if b.key == "inv_start"]


def test_missing_key_setup_overrides_a_local_env_file(ui_env, monkeypatch, tmp_path) -> None:
    # Regression: a developer's .env with a real key must not leak into the "no key" scenario.
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_value_from_dotenv_file\n")
    monkeypatch.chdir(tmp_path)
    ui_env(with_key=False)
    assert get_settings().groq_api_key.get_secret_value() == ""


def test_start_controls_require_consent_for_masked_samples(ui_env) -> None:
    ui_env(with_key=True)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="sample_choice").select("Customer orders (CSV)").run()
    assert not at.exception
    assert at.radio(key="inv_privacy").value == "metadata_only"
    assert at.button(key="inv_start").disabled is False
    at.radio(key="inv_privacy").set_value("masked_samples").run()
    assert at.button(key="inv_start").disabled is True
    at.checkbox(key="inv_consent").check().run()
    assert at.button(key="inv_start").disabled is False


def test_resume_after_restart_and_approve_through_ui(ui_env, make_service, sample_bytes) -> None:
    # A run paused at approval by a previous "process" (scripted LLMs).
    service, _ = make_service({"data_investigator": sc.investigator_script(),
                               "root_cause_analyst": sc.root_cause_script(),
                               "remediation_agent": sc.remediation_script()})
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    assert service.start(run_id).status == "awaiting_approval"
    store = service.services.store
    service.close()

    ui_env(with_key=True)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key="inv_pick").select(run_id).run()
    at.button(key="inv_open").click().run()
    assert not at.exception, at.exception
    assert at.text_input(key="inv_reviewer") is not None

    at.text_input(key="inv_reviewer").input("alice")
    at.radio(key="inv_decision_FIX-1").set_value("approve")
    next(b for b in at.button if b.label == "Submit decisions").click().run()
    assert not at.exception, at.exception

    assert store.load_manifest(run_id).artifacts == ["remediated"]
    assert any("Completed" in str(m.value) for m in at.markdown)
    entries = AuditLog(store.audit_path(run_id)).read()
    assert [e["event"] for e in entries] == ["approval_decision", "execution_started", "fix_execution", "artifact_saved"]
    assert entries[0]["reviewer"] == "alice" and entries[2]["status"] == "executed"


def test_stale_decision_submission_shows_notice_not_exception(ui_env, make_service, sample_bytes) -> None:
    # Another session decides first; this session's (stale) form submission must be refused gracefully.
    service, _ = make_service({"data_investigator": sc.investigator_script(),
                               "root_cause_analyst": sc.root_cause_script(),
                               "remediation_agent": sc.remediation_script()})
    run_id = service.create_run(sample_bytes, "customer_orders.csv").run_id
    snap = service.start(run_id)
    ui_env(with_key=True)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key="inv_pick").select(run_id).run()
    at.button(key="inv_open").click().run()
    at.text_input(key="inv_reviewer").input("alice")
    at.radio(key="inv_decision_FIX-1").set_value("approve")

    fix = snap.pending_approval["fixes"][0]
    service.resume(run_id, [{"fix_id": fix["fix_id"], "sql_sha256": fix["sql_sha256"], "decision": "rejected",
                             "reviewer": "bob"}])  # the other session wins
    next(b for b in at.button if b.label == "Submit decisions").click().run()

    assert not at.exception, at.exception
    # The page re-reads the checkpoint first: the stale form is gone and the decided run is shown instead.
    assert not [t for t in at.text_input if t.key == "inv_reviewer"]
    assert any("Completed" in str(m.value) for m in at.markdown)
    assert service.services.store.load_manifest(run_id).artifacts == []  # alice's stale approval did nothing
    events = [e["event"] for e in AuditLog(service.services.store.audit_path(run_id)).read()]
    assert events == ["approval_decision"]
