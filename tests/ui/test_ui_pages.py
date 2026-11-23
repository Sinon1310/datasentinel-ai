"""Tests for the redesigned shell, view-model helpers and the Pipeline Observability page."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import streamlit as st

from app.components import ui
from app.components.shell import MAIN_SECTIONS, NAV_ITEMS, OBSERVABILITY_PAGE
from app.views import overview, quality, schema
from datasentinel.config import get_settings
from datasentinel.pipeline import analyze_file

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app" / "streamlit_app.py")
SAMPLE = ROOT / "sample_data" / "customer_orders.csv"


@pytest.fixture(scope="module")
def result():
    return analyze_file(SAMPLE)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Point the app at an empty data dir with no API key (offline guard stays active)."""
    monkeypatch.setenv("DATASENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("DATASENTINEL_GROQ_API_KEY", "")
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield tmp_path / "data"
    get_settings.cache_clear()
    st.cache_resource.clear()


# ---------------------------------------------------------------------- formatting and view models


def test_formatters_never_fabricate_values() -> None:
    assert ui.fmt_score(None) == "—" and ui.fmt_int(None) == "—" and ui.fmt_pct(None) == "—"
    assert ui.fmt_duration_ms(None) == "—" and ui.fmt_duration_ms(450) == "450 ms"
    assert ui.fmt_duration_ms(90_000) == "1m 30s" and ui.fmt_duration_ms(5_400_000) == "1h 30m"
    assert ui.fmt_score(0.0) == "0.0"  # a real zero is still shown as zero


def test_quality_kpis_preserve_unscored_semantics(result) -> None:
    unscored = dataclasses.replace(result, quality=dataclasses.replace(result.quality, overall_score=None, grade="N/A"))
    kpis = {k.label: k for k in quality.quality_kpis(unscored)}
    assert kpis["Quality score"].value == "—" and "No check was evaluated" in kpis["Quality score"].help
    assert kpis["Grade"].value == "N/A"
    scored = {k.label: k.value for k in quality.quality_kpis(result)}
    assert scored["Quality score"] == f"{result.quality.overall_score:.1f}" and scored["Grade"] == result.quality.grade


def test_quality_kpis_count_from_report(result) -> None:
    kpis = {k.label: k.value for k in quality.quality_kpis(result)}
    summary = result.quality.summary()
    assert kpis["Passed checks"] == str(summary["by_status"]["passed"])
    assert kpis["Failed checks"] == str(summary["by_status"]["failed"])
    assert kpis["Critical issues"] == str(summary["failed_by_severity"]["critical"])


def test_results_table_uses_engine_fields_and_generic_guidance_only_on_failures(result) -> None:
    table = quality.results_table(result)
    assert len(table) == len(result.quality.results)
    assert set(table.columns) >= {"check", "column", "severity", "status", "affected_rows", "description", "guidance"}
    assert (table.loc[table["status"] != "failed", "guidance"] == "").all()
    assert (table.loc[table["status"] == "failed", "guidance"] != "").all()


def test_overview_kpis_and_actions_are_deterministic(result) -> None:
    kpis = {k.label: k.value for k in overview.overview_kpis(result)}
    assert kpis["Rows"] == f"{result.profile.n_rows:,}" and kpis["Duplicate rows"] == str(result.profile.duplicate_rows)
    actions = overview.next_actions(result, has_baseline=False, ai_configured=False)
    assert any("schema baseline" in a for a in actions) and any("GROQ_API_KEY" in a for a in actions)
    assert actions == overview.next_actions(result, has_baseline=False, ai_configured=False)


def test_schema_contract_status_reflects_engine_results(result, tmp_path) -> None:
    assert schema.contract_status(result) == "No contract"
    from datasentinel.schema import SchemaContract

    contract = SchemaContract.from_dict({"columns": {"order_id": {"type": "numeric", "unique": True},
                                                     "missing_col": {"type": "text"}}})
    with_contract = analyze_file(SAMPLE, contract=contract)
    assert schema.contract_status(with_contract) == "2 violation(s)"
    assert schema.column_contract_map(with_contract)["order_id"] == "failed"


def test_navigation_model_has_six_destinations_and_one_page() -> None:
    assert [i.label for i in NAV_ITEMS] == ["Overview", "Data Quality", "Schema Explorer", "Data Profiling",
                                            "AI Investigation", "Pipeline Observability"]
    assert MAIN_SECTIONS == ("overview", "quality", "schema", "profiling", "investigation")
    assert (ROOT / "app" / OBSERVABILITY_PAGE).is_file()


# ---------------------------------------------------------------------- app shell


def test_unknown_section_falls_back_to_overview(isolated) -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["section"] = "<script>alert(1)</script>"
    at.run()
    assert not at.exception and [t.value for t in at.title] == ["Overview"]


def test_empty_state_on_every_section(isolated) -> None:
    for section in MAIN_SECTIONS:
        at = AppTest.from_file(APP, default_timeout=60)
        at.query_params["section"] = section
        at.run()
        assert not at.exception, (section, at.exception)
        assert not at.metric or section == "investigation", section  # no fabricated KPIs without data


def test_unscored_report_renders_dash_and_na() -> None:
    at = AppTest.from_string(
        "import sys, dataclasses\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from datasentinel.pipeline import analyze_file\n"
        "from app.components.dataset import AnalysisState, ActiveDataset\n"
        "from app.views import quality\n"
        f"r = analyze_file({str(SAMPLE)!r})\n"
        "r = dataclasses.replace(r, quality=dataclasses.replace(r.quality, overall_score=None, grade='N/A'))\n"
        "quality.render(AnalysisState(ActiveDataset(b'', 'x.csv', 'sample', 'now', 'abc'), r))\n",
        default_timeout=60).run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Quality score"] == "—" and values["Grade"] == "N/A"


def test_dataset_survives_navigation_to_observability_and_back(isolated) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="sample_choice").select("Customer orders (CSV)").run()
    at.switch_page(OBSERVABILITY_PAGE).run()
    assert not at.exception
    at.switch_page("streamlit_app.py").run()
    assert {m.label for m in at.metric} >= {"Rows", "Quality score"}
    at.button(key="clear_dataset").click().run()
    assert not at.metric  # cleared explicitly


# ---------------------------------------------------------------------- pipeline observability page


def _open_observability(isolated):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.switch_page(OBSERVABILITY_PAGE).run()
    return at


def test_observability_defaults_to_not_connected_without_fake_data(isolated) -> None:
    at = _open_observability(isolated)
    assert not at.exception
    assert not at.metric  # no KPIs shown when no real backend is connected
    assert any("not connected" in m.value for m in at.markdown)
    assert [b.key for b in at.button if b.key] == ["obs_try_demo"]


def test_demo_mode_is_labelled_and_renders_all_sections(isolated) -> None:
    at = _open_observability(isolated)
    at.button(key="obs_try_demo").click().run()
    assert not at.exception, at.exception
    assert any("Synthetic demo data" in m.value for m in at.markdown)
    assert {m.label for m in at.metric} >= {"Total executions", "Success rate", "Failed executions",
                                            "Running executions", "Average duration", "Processed rows"}
    from app.observability.contracts import RunFilters

    assert at.session_state["obs_demo_backend"].summary(RunFilters()).total_runs > 0
    assert not isolated.exists() or not list(isolated.glob("**/*"))  # nothing persisted anywhere


def test_run_detail_actions_follow_backend_rules(isolated) -> None:
    from app.observability.contracts import RunFilters, RunStatus

    at = _open_observability(isolated)
    at.button(key="obs_try_demo").click().run()
    at.selectbox(key="obs_window").select("All time").run()
    backend = at.session_state["obs_demo_backend"]
    runs = backend.list_runs(RunFilters(), limit=200).items
    cases = {
        "pending": next(r for r in runs if r.status == RunStatus.PENDING),
        "success": next(r for r in runs if r.status == RunStatus.SUCCESS),
    }
    for name, run in cases.items():
        at.selectbox(key="obs_run").select(run.pipeline_run_id).run()  # raw option value (formatted for display)
        assert not at.exception, at.exception
        retry, cancel = at.button(key="obs_retry_btn"), at.button(key="obs_cancel_btn")
        if name == "pending":
            assert retry.disabled and not cancel.disabled
        else:
            assert retry.disabled and cancel.disabled


def test_retry_dialog_opens_with_required_inputs(isolated) -> None:
    from app.observability.contracts import RunFilters

    at = _open_observability(isolated)
    at.button(key="obs_try_demo").click().run()
    at.selectbox(key="obs_window").select("All time").run()
    backend = at.session_state["obs_demo_backend"]
    retryable = next(r for r in backend.list_runs(RunFilters(), limit=200).items
                     if backend.get_run(r.pipeline_run_id).allowed_actions.can_request_retry)
    at.selectbox(key="obs_run").select(retryable.pipeline_run_id).run()
    assert not at.button(key="obs_retry_btn").disabled
    at.button(key="obs_retry_btn").click().run()
    assert {t.key for t in at.text_input} >= {"obs_retry_who"}
    assert at.button(key="obs_retry_confirm").disabled  # requires a name and a reason


def _all_text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def test_connection_state_is_never_overstated(isolated) -> None:
    at = _open_observability(isolated)
    text = _all_text(at)
    assert "Observability backend: not connected" in text and "Showing: no data" in text
    assert "backend: connected" not in text
    assert at.button_group[0].options[0] == "Live data (not connected)"  # selector label states availability
    at.button(key="obs_try_demo").click().run()
    text = _all_text(at)
    # Viewing demo data does not change backend availability, and the two facts are shown separately.
    assert "Observability backend: not connected" in text and "Showing: synthetic demo data (in memory)" in text
    assert "backend: connected" not in text


def test_table_cells_never_show_none() -> None:
    assert ui.display_value(None) == "—" and ui.display_value(float("nan")) == "—"
    assert ui.display_value(2500000) == "2,500,000" and ui.display_value(0) == "0"
    assert ui.display_value(True) == "yes" and ui.display_value("x") == "x"
