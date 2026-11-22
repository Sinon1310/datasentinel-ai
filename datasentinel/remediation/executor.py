"""Approval-gated execution of remediation SQL.

This is the ONLY code path that applies fixes, and it accepts only fixes carrying an
approved SHA-256. Immediately before running each fix it re-computes the hash of the SQL
it is about to execute and re-validates the policy; any mismatch blocks that fix. Fixes
are applied in order to a copy of the data and the result is returned as a new DataFrame —
the original dataset is never modified or overwritten. Every decision is audited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

from datasentinel.remediation.audit import AuditLog
from datasentinel.remediation.sandbox import DuckDBSandbox, SandboxError
from datasentinel.remediation.sql_guard import sql_sha256, validate_sql

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovedFix:
    """A fix as approved by a human: the exact SQL and the hash they approved."""

    fix_id: str
    sql: str
    approved_sha256: str
    approved_by: str


@dataclass
class FixExecutionRecord:
    """Outcome of one fix."""

    fix_id: str
    status: str  # executed | blocked | failed
    reason: str
    sql_sha256: str
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None


@dataclass
class ExecutionOutcome:
    """Outcome of an execution batch."""

    records: List[FixExecutionRecord] = field(default_factory=list)
    output: Optional[pd.DataFrame] = None

    @property
    def executed_ids(self) -> List[str]:
        """Ids of fixes that were applied."""
        return [r.fix_id for r in self.records if r.status == "executed"]


def execute_approved_fixes(
    df: pd.DataFrame,
    fixes: Sequence[ApprovedFix],
    sandbox: Optional[DuckDBSandbox] = None,
    audit: Optional[AuditLog] = None,
) -> ExecutionOutcome:
    """Apply approved fixes sequentially to a copy of ``df``.

    Returns:
        An :class:`ExecutionOutcome`; ``output`` is ``None`` if no fix executed.
    """
    sandbox = sandbox or DuckDBSandbox()
    current = df.copy(deep=True)
    original_columns = set(df.columns)
    outcome = ExecutionOutcome()

    def _log(record: FixExecutionRecord) -> None:
        outcome.records.append(record)
        if audit:
            audit.record("fix_execution", fix_id=record.fix_id, status=record.status, reason=record.reason,
                         sql_sha256=record.sql_sha256, rows_before=record.rows_before, rows_after=record.rows_after)

    for fix in fixes:
        actual_hash = sql_sha256(fix.sql)
        if actual_hash != fix.approved_sha256:
            _log(FixExecutionRecord(fix.fix_id, "blocked", "SQL hash does not match the approved hash", actual_hash))
            continue
        validation = validate_sql(fix.sql)
        if not validation.ok:
            _log(FixExecutionRecord(fix.fix_id, "blocked", "Policy re-validation failed: " + "; ".join(validation.violations), actual_hash))
            continue
        rows_before = len(current)
        try:
            output = sandbox.run(fix.sql, current)
        except SandboxError as exc:
            _log(FixExecutionRecord(fix.fix_id, "failed", str(exc), actual_hash, rows_before))
            continue
        if set(output.columns) != original_columns:
            _log(FixExecutionRecord(fix.fix_id, "blocked", "Output columns differ from the dataset's columns", actual_hash, rows_before, len(output)))
            continue
        current = output[list(df.columns)]
        _log(FixExecutionRecord(fix.fix_id, "executed", f"Approved by {fix.approved_by}", actual_hash, rows_before, len(current)))

    if outcome.executed_ids:
        outcome.output = current.reset_index(drop=True)
    logger.info("Executed %d of %d approved fix(es)", len(outcome.executed_ids), len(fixes))
    return outcome
