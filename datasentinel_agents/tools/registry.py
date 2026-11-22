"""Per-agent tool allow-lists (least privilege).

No agent receives an execution tool. Execution happens only in the deterministic
``execute_fixes`` node after human approval.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from datasentinel_agents.tools import library as lib
from datasentinel_agents.tools.base import ToolContext, make_tool

# name -> (description, args schema, implementation)
_CATALOG = {
    "get_quality_summary": ("Overall quality score, grade, dimension scores and check counts.", lib.NoArgs, lib.quality_summary),
    "list_failed_checks": ("Failed quality checks, optionally filtered by dimension or severity.", lib.FailedChecksArgs, lib.list_failed_checks),
    "get_check_details": ("Full evidence for one quality check result (bounds, counts, value shapes).", lib.CheckArgs, lib.check_details),
    "get_column_schema": ("Inferred schema of one column, including parse ratio and notes.", lib.ColumnArgs, lib.column_schema),
    "get_column_profile": ("Statistical profile of one column.", lib.ColumnArgs, lib.column_profile),
    "get_inferred_schema": ("All columns with inferred types, nullability and primary-key candidates.", lib.NoArgs, lib.inferred_schema),
    "sample_issue_rows": ("Row indices and privacy-filtered value descriptions for rows with an issue.", lib.SampleArgs, lib.sample_issue_rows),
    "issue_rate_by_segment": ("Issue rate per value of a low-cardinality column, with lift vs overall.", lib.SegmentArgs, lib.issue_rate_by_segment),
    "issue_trend": ("Issue counts and rates per time period of a datetime column.", lib.TrendArgs, lib.issue_trend),
    "missingness_cooccurrence": ("How often columns are missing together.", lib.CooccurrenceArgs, lib.missingness_cooccurrence),
    "get_schema_diff": ("Deterministic diff between the baseline schema and the current schema.", lib.NoArgs, lib.schema_diff),
    "get_baseline_contract": ("The baseline schema contract (expected types, nullability, uniqueness).", lib.NoArgs, lib.baseline_contract),
    "find_rename_candidates": ("Removed/added column pairs that look like renames.", lib.NoArgs, lib.rename_candidates),
    "get_evidence": ("Look up an evidence record by id.", lib.EvidenceArgs, lib.get_evidence),
    "list_findings": ("Findings produced so far, with status and evidence ids.", lib.NoArgs, lib.list_findings),
    "dry_run_sql": ("Validate a fix (SQL policy + optional PySpark static check) and dry-run it on a sample. Never modifies data.", lib.SqlArgs, lib.dry_run_sql),
}

AGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "data_investigator": (
        "get_quality_summary", "list_failed_checks", "get_check_details", "get_column_schema", "get_column_profile",
        "get_inferred_schema", "sample_issue_rows", "issue_rate_by_segment", "issue_trend", "missingness_cooccurrence",
    ),
    "schema_analyst": (
        "get_schema_diff", "get_baseline_contract", "get_inferred_schema", "get_column_schema", "get_column_profile",
        "find_rename_candidates",
    ),
    "root_cause_analyst": (
        "get_evidence", "list_findings", "get_quality_summary", "issue_rate_by_segment", "issue_trend",
        "missingness_cooccurrence",
    ),
    "remediation_agent": (
        "get_check_details", "get_column_schema", "get_column_profile", "get_inferred_schema", "list_findings",
        "get_evidence", "dry_run_sql",
    ),
}

_UNRECORDED = {"get_evidence", "list_findings"}  # reads of the ledger itself create no new evidence


def build_tools(agent: str, ctx: ToolContext) -> list[BaseTool]:
    """Instantiate the tools an agent is allowed to use."""
    tools = []
    for name in AGENT_TOOLS[agent]:
        description, schema, impl = _CATALOG[name]
        tools.append(make_tool(ctx, name, description, schema, impl, record=name not in _UNRECORDED))
    return tools
