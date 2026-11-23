"""Architecture guard rails enforced as tests."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IN_ENGINE = ("datasentinel_agents", "langchain", "langchain_core", "langchain_groq", "langgraph", "groq",
                       "streamlit", "app")


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_engine_never_imports_llm_ui_or_agent_code() -> None:
    offenders = {}
    for path in (ROOT / "datasentinel").rglob("*.py"):
        bad = _imports(path) & set(FORBIDDEN_IN_ENGINE)
        if bad:
            offenders[str(path.relative_to(ROOT))] = sorted(bad)
    assert offenders == {}, f"Deterministic engine must not import LLM/UI/agent code: {offenders}"


def test_agents_never_import_streamlit() -> None:
    offenders = [str(p.relative_to(ROOT)) for p in (ROOT / "datasentinel_agents").rglob("*.py")
                 if {"streamlit", "app"} & _imports(p)]
    assert offenders == []


def test_no_agent_module_executes_generated_code() -> None:
    for path in list((ROOT / "datasentinel_agents").rglob("*.py")) + list((ROOT / "datasentinel").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile"}, f"{path}: {node.func.id}() is forbidden"


def test_ui_has_no_direct_persistence_or_execution_access() -> None:
    """The UI reaches data only through services/contracts: no raw SQLite, no observability writers,
    no direct execution of remediation SQL (approval-gated execution stays in the agent service)."""
    forbidden_modules = {"sqlite3", "duckdb"}
    forbidden_names = ("PipelineTracker", "ObservabilityStore", "execute_approved_fixes", "DuckDBSandbox", "SqliteSaver")
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text()
        bad = (_imports(path) & forbidden_modules) | {n for n in forbidden_names if n in text}
        if bad:
            offenders.append((str(path.relative_to(ROOT)), sorted(bad)))
    assert offenders == [], offenders


def test_observability_ui_contracts_are_independent_of_agents_and_engine_writers() -> None:
    for path in (ROOT / "app" / "observability").rglob("*.py"):
        assert not ({"datasentinel_agents", "langchain", "langgraph", "groq"} & _imports(path)), path
