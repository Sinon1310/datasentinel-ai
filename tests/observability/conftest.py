"""Fixtures for the pipeline observability tests. Every database lives in a pytest temporary directory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Iterator

import pytest

from datasentinel.observability import (
    CategorySource, FailureCategory, FailureRecord, FixedClock, ObservabilityStore, PipelineRun, PipelineSpec,
    TriggerType,
)
from datasentinel.observability.models import _seal_sanitized

# Test-only stand-in for the Phase 2 redactor: these literals contain nothing sensitive.
S = _seal_sanitized
ACTOR = S("test.runner")
SPEC = PipelineSpec("orders.daily", "Orders daily load")


def failure(primary: bool = True, *, task_id: str | None = None, message: str = "boom",
            category: FailureCategory = FailureCategory.TRANSFORMATION) -> FailureRecord:
    return FailureRecord(category=category, category_source=CategorySource.CLASSIFIER, classifier_rule="test.rule",
                         classifier_version="v1", error_type="builtins.ValueError", message=S(message),
                         is_primary=primary, pipeline_task_run_id=task_id)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(start_ms=1_800_000_000_000)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "obs" / "observability.sqlite"


@pytest.fixture
def store(db_path: Path, clock: FixedClock) -> ObservabilityStore:
    return ObservabilityStore(db_path, clock=clock)


@pytest.fixture
def new_run(store: ObservabilityStore) -> Callable[..., PipelineRun]:
    def _make(start: bool = True, spec: PipelineSpec = SPEC) -> PipelineRun:
        trigger = TriggerType.SYNTHETIC if spec.is_synthetic else TriggerType.MANUAL
        return store.create_run(spec, actor=ACTOR, trigger_type=trigger, start=start)
    return _make


@pytest.fixture
def raw(db_path: Path, store: ObservabilityStore) -> Iterator[sqlite3.Connection]:
    """A raw connection that bypasses the store entirely (used to prove database-level enforcement)."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()
