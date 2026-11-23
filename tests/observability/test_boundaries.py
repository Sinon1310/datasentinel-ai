"""The observability package stays inside the deterministic engine boundary and adds no dependencies."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "datasentinel" / "observability"


def _top_level_imports(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_observability_uses_only_the_standard_library_and_the_engine() -> None:
    allowed = set(sys.stdlib_module_names) | {"datasentinel", "__future__"}
    for path in PACKAGE.rglob("*.py"):
        extra = _top_level_imports(path) - allowed
        assert not extra, f"{path.name} imports non-stdlib modules: {sorted(extra)}"


def test_only_phase_one_and_two_modules_exist() -> None:
    assert sorted(p.name for p in PACKAGE.glob("*.py")) == [
        "__init__.py", "classifier.py", "clock.py", "errors.py", "gates.py", "lifecycle.py", "metrics.py",
        "migrations.py", "models.py", "redaction.py", "store.py", "tracker.py"]
