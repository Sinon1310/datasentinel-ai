"""Versioned system prompts (Markdown files alongside this module)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent
AGENTS = ("data_investigator", "schema_analyst", "root_cause_analyst", "remediation_agent")


@lru_cache(maxsize=None)
def load_prompt(agent: str) -> str:
    """The full system prompt for ``agent``: role prompt followed by shared operating rules."""
    if agent not in AGENTS:
        raise KeyError(agent)
    return (_DIR / f"{agent}.md").read_text(encoding="utf-8") + "\n\n" + (_DIR / "common.md").read_text(encoding="utf-8")
