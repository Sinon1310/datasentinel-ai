"""Headless smoke tests for the Streamlit dashboard using streamlit.testing."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_welcome_screen_renders() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Upload a CSV or JSON" in i.value for i in at.info)


SECTION_TITLES = {
    "overview": "Overview", "quality": "Data Quality", "schema": "Schema Explorer",
    "profiling": "Data Profiling", "investigation": "AI Investigation",
}


@pytest.mark.parametrize("sample", ["Customer orders (CSV)", "Sensor readings (nested JSON)"])
def test_sample_dataset_renders_all_sections(sample: str) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="sample_choice").select(sample).run()
    assert not at.exception, at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Quality score" in labels and "Rows" in labels
    # Every section of the redesigned navigation is reachable and renders without errors
    # (replaces the former six-tab layout).
    for section, title in SECTION_TITLES.items():
        at.query_params["section"] = section
        at.run()
        assert not at.exception, (section, at.exception)
        assert not at.error, (section, [e.value for e in at.error])
        assert title in [t.value for t in at.title], section
    at.switch_page("pages/1_Pipeline_Observability.py").run()
    assert not at.exception, at.exception
    assert "Pipeline Observability" in [t.value for t in at.title]
