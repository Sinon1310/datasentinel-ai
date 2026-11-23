"""Observability settings: defaults keep existing behaviour, values are validated, nothing is created eagerly."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from datasentinel.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_match_the_approved_design() -> None:
    s = _settings()
    assert (s.observability_db, s.observability_demo_db) == ("observability.sqlite", "observability_demo.sqlite")
    assert (s.pipeline_stale_after_seconds, s.pipeline_heartbeat_min_interval_seconds) == (900, 15)
    assert (s.quality_gate_min_score, s.quality_gate_fail_on_critical, s.quality_gate_fail_when_unscored) == \
           (60.0, True, True)
    assert s.observability_db_path == Path(".datasentinel") / "observability.sqlite"
    assert s.observability_demo_db_path == Path(".datasentinel") / "observability_demo.sqlite"


def test_environment_overrides_use_the_existing_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATASENTINEL_QUALITY_GATE_MIN_SCORE", "75")
    monkeypatch.setenv("DATASENTINEL_PIPELINE_STALE_AFTER_SECONDS", "120")
    monkeypatch.setenv("DATASENTINEL_OBSERVABILITY_DB", "pipelines.sqlite")
    monkeypatch.setenv("DATASENTINEL_DATA_DIR", str(tmp_path))
    s = _settings()
    assert (s.quality_gate_min_score, s.pipeline_stale_after_seconds) == (75.0, 120)
    assert s.observability_db_path == tmp_path / "pipelines.sqlite"


@pytest.mark.parametrize("name", ["", "..", "../escape.sqlite", "/abs.sqlite", "sub/dir.sqlite", "a\\b.sqlite",
                                  "file:obs.sqlite", ":memory:", ".hidden.sqlite", "bad\nname"])
def test_database_names_must_be_plain_file_names(name: str) -> None:
    with pytest.raises(ValidationError):
        _settings(observability_db=name)
    with pytest.raises(ValidationError):
        _settings(observability_demo_db=name)


@pytest.mark.parametrize("field,value", [("quality_gate_min_score", -0.1), ("quality_gate_min_score", 100.1),
                                         ("pipeline_stale_after_seconds", 0),
                                         ("pipeline_heartbeat_min_interval_seconds", -5)])
def test_numeric_settings_are_bounded(field: str, value) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_loading_settings_creates_no_files(tmp_path: Path) -> None:
    s = _settings(data_dir=tmp_path / "never-created")
    assert s.observability_db_path.name == "observability.sqlite"
    assert not (tmp_path / "never-created").exists()
