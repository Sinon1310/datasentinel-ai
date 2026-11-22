"""Evidence ledger: seeding from engine output and recording tool results.

Id scheme (unique across parallel branches):

* ``E-Q-NNN``   failed quality checks (seeded by ``prepare_context``)
* ``E-S-NNN``   deterministic schema-diff items (seeded)
* ``E-INV-NNN`` / ``E-SCH-NNN`` / ``E-RCA-NNN`` / ``E-REM-NNN``  tool calls per agent
"""

from __future__ import annotations

import math
import re
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from datasentinel.investigation.privacy import safe_identifier
from datasentinel.investigation.workspace import Workspace
from datasentinel.quality.models import CheckStatus
from datasentinel.utils.serialization import to_builtin
from datasentinel_agents.models import Evidence

AGENT_PREFIXES = {
    "data_investigator": "INV",
    "schema_analyst": "SCH",
    "root_cause_analyst": "RCA",
    "remediation_agent": "REM",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_index(evidence: Iterable[Evidence], prefix: str) -> int:
    """Next free sequence number for ``E-<prefix>-NNN`` ids."""
    pattern = re.compile(rf"^E-{re.escape(prefix)}-(\d+)$")
    numbers = [int(m.group(1)) for e in evidence if (m := pattern.match(e.evidence_id))]
    return max(numbers, default=0) + 1


class EvidenceRecorder:
    """Creates evidence records for one agent run (thread-safe)."""

    def __init__(self, agent: str, start: int = 1) -> None:
        self.agent = agent
        self.prefix = AGENT_PREFIXES[agent]
        self._next = start
        self._lock = threading.Lock()
        self.records: list[Evidence] = []

    def record(self, tool: str, arguments: dict[str, Any], summary: str, data: dict[str, Any]) -> Evidence:
        """Append a tool-result evidence record and return it."""
        with self._lock:
            evidence = Evidence(
                evidence_id=f"E-{self.prefix}-{self._next:03d}",
                source="tool",
                tool=tool,
                arguments=to_builtin(arguments),
                summary=summary,
                data=to_builtin(data),
                produced_by=self.agent,
                created_at=_now(),
            )
            self._next += 1
            self.records.append(evidence)
        return evidence


def seed_evidence(workspace: Workspace) -> tuple[list[Evidence], dict[tuple[str, str | None], str]]:
    """Turn failed quality checks and schema-diff items into evidence.

    Returns:
        ``(evidence, check_index)`` where ``check_index`` maps ``(rule_id, column)`` to evidence ids.
    """
    guard = workspace.guard
    evidence: list[Evidence] = []
    index: dict[tuple[str, str | None], str] = {}
    created = _now()
    failed = [r for r in workspace.result.quality.results if r.status == CheckStatus.FAILED]
    for number, result in enumerate(failed, start=1):
        evidence_id = f"E-Q-{number:03d}"
        index[(result.rule_id, result.column)] = evidence_id
        details = {k: v for k, v in to_builtin(result.details).items() if k != "sample_row_indices"}
        evidence.append(Evidence(
            evidence_id=evidence_id,
            source="quality_engine",
            tool="quality_engine",
            arguments={"rule_id": result.rule_id, "column": result.column},
            summary=f"{result.rule_name} failed on {safe_identifier(result.target)}: {result.message}",
            data={
                "rule_id": result.rule_id,
                "column": safe_identifier(result.column) if result.column else None,
                "dimension": result.dimension.value,
                "severity": result.severity.value,
                "affected_rows": result.affected_rows,
                "affected_pct": round(result.affected_pct, 4),
                "threshold": result.threshold,
                "details": details,
                "sample_values": guard.describe_values(result.sample_values),
            },
            produced_by="system",
            created_at=created,
        ))
    if workspace.schema_diff is not None:
        for number, change in enumerate(workspace.schema_diff.changes, start=1):
            evidence.append(Evidence(
                evidence_id=f"E-S-{number:03d}",
                source="drift_engine",
                tool="schema_drift",
                arguments={"change_type": change.change_type, "column": change.column},
                summary=f"{change.change_type} on {safe_identifier(change.column)} ({change.compatibility}): {change.reason}",
                data={**to_builtin(change), "column": safe_identifier(change.column)},
                produced_by="system",
                created_at=created,
            ))
    return evidence, index


def numbers_in(value: Any) -> set[float]:
    """All finite numbers contained (recursively) in a JSON-like value."""
    found: set[float] = set()
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            found.add(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            found |= numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= numbers_in(item)
    return found
