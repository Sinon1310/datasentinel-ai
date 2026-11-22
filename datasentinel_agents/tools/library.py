"""Tool implementations. Each is a thin, read-only adapter over the deterministic engine.

No statistics are computed here beyond formatting: quality results come from
``QualityReport``, profiles from ``DatasetProfile``, drill-downs from
``datasentinel.investigation.drilldown``, drift from ``datasentinel.schema.drift`` and SQL
checks from ``datasentinel.remediation``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from datasentinel.investigation import drilldown
from datasentinel.investigation.privacy import safe_identifier
from datasentinel.quality.models import CheckStatus
from datasentinel.remediation import SandboxError, check_pyspark, validate_sql
from datasentinel.utils.serialization import to_builtin
from datasentinel_agents.tools.base import ToolContext, ToolInputError

IssueLiteral = Literal["missing", "invalid_type", "invalid_date", "outlier", "duplicate"]


# ---------------------------------------------------------------------- argument schemas


class NoArgs(BaseModel):
    """No arguments."""


class ColumnArgs(BaseModel):
    column: str = Field(max_length=200, description="Exact column name.")


class FailedChecksArgs(BaseModel):
    dimension: Literal["completeness", "uniqueness", "validity", "consistency", "accuracy"] | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None


class CheckArgs(BaseModel):
    rule_id: str = Field(max_length=100, description="e.g. 'completeness.missing_values'.")
    column: str | None = Field(default=None, max_length=200, description="Column, or null for dataset-level checks.")


class IssueArgs(BaseModel):
    column: str | None = Field(default=None, max_length=200, description="Column (null only for issue='duplicate').")
    issue: IssueLiteral


class SampleArgs(IssueArgs):
    limit: int = Field(default=10, ge=1, le=20)


class SegmentArgs(IssueArgs):
    by_column: str = Field(max_length=200, description="Low-cardinality column to segment by.")
    top_n: int = Field(default=10, ge=1, le=25)


class TrendArgs(IssueArgs):
    date_column: str = Field(max_length=200, description="A datetime column.")
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"


class CooccurrenceArgs(BaseModel):
    columns: list[str] = Field(min_length=2, max_length=8)


class EvidenceArgs(BaseModel):
    evidence_id: str = Field(max_length=30)


class SqlArgs(BaseModel):
    sql: str = Field(max_length=10_000, description="DuckDB SELECT over `dataset` returning all columns.")
    pyspark: str | None = Field(default=None, max_length=10_000, description="Optional equivalent PySpark to statically check.")


# ---------------------------------------------------------------------- helpers


def _column(ctx: ToolContext, column: str) -> str:
    if column not in ctx.workspace.result.raw_dataframe.columns:
        raise ToolInputError(f"Unknown column '{safe_identifier(column)}'. Use get_inferred_schema to list columns.")
    return column


# ---------------------------------------------------------------------- quality


def quality_summary(ctx: ToolContext) -> tuple[str, dict[str, Any]]:
    quality = ctx.workspace.result.quality
    summary = quality.summary()
    data = {
        "overall_score": quality.overall_score,
        "grade": quality.grade,
        "dimensions": {d.dimension.value: {"score": d.score, "effective_weight": d.effective_weight,
                                           "checks": d.checks_evaluated, "failed": d.checks_failed}
                       for d in quality.dimension_scores},
        "checks": summary,
    }
    return f"Quality score {quality.overall_score} ({quality.grade}); {summary['by_status']['failed']} failed checks", data


def list_failed_checks(ctx: ToolContext, dimension: str | None = None, severity: str | None = None) -> tuple[str, dict]:
    items = []
    for r in ctx.workspace.result.quality.failed_results:
        if (dimension and r.dimension.value != dimension) or (severity and r.severity.value != severity):
            continue
        items.append({"rule_id": r.rule_id, "column": safe_identifier(r.column) if r.column else None,
                      "severity": r.severity.value, "affected_rows": r.affected_rows,
                      "affected_pct": round(r.affected_pct, 4), "message": r.message})
    return f"{len(items)} failed check(s)", {"failed_checks": items}


def check_details(ctx: ToolContext, rule_id: str, column: str | None = None) -> tuple[str, dict]:
    match = next((r for r in ctx.workspace.result.quality.results if r.rule_id == rule_id and r.column == column), None)
    if match is None:
        raise ToolInputError(f"No result for rule '{rule_id}' on column {column!r}. Use list_failed_checks.")
    details = {k: v for k, v in to_builtin(match.details).items() if k != "sample_row_indices"}
    data = {"rule_id": rule_id, "column": safe_identifier(column) if column else None, "status": match.status.value,
            "severity": match.severity.value, "affected_rows": match.affected_rows,
            "affected_pct": round(match.affected_pct, 4), "threshold": match.threshold, "message": match.message,
            "details": details, "sample_values": ctx.workspace.guard.describe_values(match.sample_values)}
    return f"{match.rule_name} on {safe_identifier(match.target)}: {match.status.value}", data


# ---------------------------------------------------------------------- schema / profile


def column_schema(ctx: ToolContext, column: str) -> tuple[str, dict]:
    col = ctx.workspace.result.schema.column(_column(ctx, column))
    data = {"name": safe_identifier(col.name), "semantic_type": col.semantic_type.value, "storage_type": col.storage_type,
            "pandas_dtype": col.pandas_dtype, "nullable": col.nullable, "null_count": col.null_count,
            "unique_count": col.unique_count, "inferred_from_string": col.inferred_from_string,
            "parse_success_ratio": col.parse_success_ratio, "notes": col.notes,
            "sample_shapes": [ctx.workspace.guard.describe_value(v)["shape"] for v in col.sample_values]}
    return f"{safe_identifier(col.name)}: {col.semantic_type.value}", data


def column_profile(ctx: ToolContext, column: str) -> tuple[str, dict]:
    name = _column(ctx, column)
    profile = ctx.workspace.result.profile.columns[name]
    guard = ctx.workspace.guard
    data: dict[str, Any] = {"name": safe_identifier(name), "semantic_type": profile.semantic_type.value,
                            "non_null_count": profile.non_null_count, "null_count": profile.null_count,
                            "null_pct": round(profile.null_pct, 4), "unique_count": profile.unique_count,
                            "is_unique": profile.is_unique, "is_constant": profile.is_constant}
    if profile.numeric:
        n = profile.numeric
        data["numeric"] = {k: getattr(n, k) for k in ("count", "min", "max", "mean", "median", "std", "q1", "q3", "iqr",
                                                      "p05", "p95", "skewness", "zeros", "negatives")}
    if profile.categorical:
        c = profile.categorical
        data["categorical"] = {
            "cardinality": c.cardinality, "mode_pct": round(c.mode_pct, 4), "normalized_entropy": c.normalized_entropy,
            "top_values": [{"label": guard.label(name, v.value), "count": v.count, "pct": round(v.pct, 4)} for v in c.top_values],
            "min_length": c.min_length, "max_length": c.max_length,
        }
    if profile.datetime:
        data["datetime"] = to_builtin(profile.datetime)
    if profile.boolean:
        data["boolean"] = to_builtin(profile.boolean)
    return f"Profile of {safe_identifier(name)}", data


def inferred_schema(ctx: ToolContext) -> tuple[str, dict]:
    schema = ctx.workspace.result.schema
    columns = [{"name": safe_identifier(c.name), "semantic_type": c.semantic_type.value, "storage_type": c.storage_type,
                "nullable": c.nullable, "null_count": c.null_count, "unique_count": c.unique_count}
               for c in schema.columns]
    keys = [pk.label for pk in schema.primary_key_candidates]
    return f"{len(columns)} columns", {"n_rows": schema.n_rows, "columns": columns, "primary_key_candidates": keys}


# ---------------------------------------------------------------------- drill-down


def sample_issue_rows(ctx: ToolContext, issue: str, column: str | None = None, limit: int = 10) -> tuple[str, dict]:
    limit = min(limit, ctx.settings.sample_row_limit)
    data = drilldown.sample_issue_rows(ctx.workspace.result, ctx.workspace.guard, column, issue, limit)
    return f"{data['matching_rows']} row(s) with {issue} in {data['column'] or 'dataset'}", data


def issue_rate_by_segment(ctx: ToolContext, issue: str, by_column: str, column: str | None = None,
                          top_n: int = 10) -> tuple[str, dict]:
    data = drilldown.issue_rate_by_segment(ctx.workspace.result, ctx.workspace.guard, column, issue, by_column, top_n)
    return f"{issue} rate by {data['by_column']} (overall {data['overall_issue_rate_pct']}%)", data


def issue_trend(ctx: ToolContext, issue: str, date_column: str, column: str | None = None,
                freq: str = "M") -> tuple[str, dict]:
    data = drilldown.issue_trend(ctx.workspace.result, column, issue, date_column, freq)
    return f"{issue} trend by {data['date_column']} ({freq})", data


def missingness_cooccurrence(ctx: ToolContext, columns: list[str]) -> tuple[str, dict]:
    data = drilldown.missingness_cooccurrence(ctx.workspace.result, columns)
    return f"Missingness co-occurrence for {len(columns)} columns", data


# ---------------------------------------------------------------------- drift


def schema_diff(ctx: ToolContext) -> tuple[str, dict]:
    diff = ctx.workspace.schema_diff
    if diff is None:
        raise ToolInputError("No baseline schema was provided for this run.")
    data = diff.to_dict()
    for change in data["changes"]:
        change["column"] = safe_identifier(change["column"])
    return f"{len(diff.changes)} schema change(s) vs {diff.baseline_source}", data


def baseline_contract(ctx: ToolContext) -> tuple[str, dict]:
    if ctx.workspace.baseline is None:
        raise ToolInputError("No baseline schema was provided for this run.")
    data = ctx.workspace.baseline.to_dict()
    data["columns"] = {safe_identifier(k): v for k, v in data["columns"].items()}
    return f"Baseline with {len(data['columns'])} columns ({ctx.workspace.baseline_source})", data


def rename_candidates(ctx: ToolContext) -> tuple[str, dict]:
    diff = ctx.workspace.schema_diff
    if diff is None:
        raise ToolInputError("No baseline schema was provided for this run.")
    items = [to_builtin(c) for c in diff.changes if c.change_type == "rename_candidate"]
    return f"{len(items)} rename candidate(s)", {"candidates": items}


# ---------------------------------------------------------------------- ledger


def get_evidence(ctx: ToolContext, evidence_id: str) -> tuple[str, dict]:
    evidence = ctx.ledger.get(evidence_id)
    if evidence is None:
        raise ToolInputError(f"Evidence '{evidence_id}' does not exist.")
    return evidence.summary, evidence.model_dump()


def list_findings(ctx: ToolContext) -> tuple[str, dict]:
    items = [{"finding_id": f.finding_id, "title": f.title, "issue_type": f.issue_type, "columns": f.columns,
              "status": f.status, "severity": f.severity, "evidence_ids": f.evidence_ids} for f in ctx.findings]
    return f"{len(items)} finding(s)", {"findings": items}


# ---------------------------------------------------------------------- remediation


def dry_run_sql(ctx: ToolContext, sql: str, pyspark: str | None = None) -> tuple[str, dict]:
    validation = validate_sql(sql)
    data: dict[str, Any] = {"sql_ok": validation.ok, "violations": validation.violations, "sql_sha256": validation.sha256}
    if pyspark is not None:
        check = check_pyspark(pyspark)
        data["pyspark_ok"], data["pyspark_violations"] = check.ok, check.violations
    if validation.ok:
        raw = ctx.workspace.result.raw_dataframe
        try:
            output = ctx.sandbox.run(sql, raw, max_input_rows=ctx.settings.dry_run_rows)
            sample_rows = min(len(raw), ctx.settings.dry_run_rows)
            data.update(dry_run_ok=True, rows_before=sample_rows, rows_after=len(output),
                        columns_preserved=set(output.columns) == set(raw.columns),
                        missing_columns=[safe_identifier(c) for c in raw.columns if c not in output.columns])
        except SandboxError as exc:
            data.update(dry_run_ok=False, error=str(exc)[:300])
    status = "valid" if validation.ok and data.get("dry_run_ok") else "invalid"
    return f"Dry run: {status}", data


def failed_check_count(ctx: ToolContext) -> int:
    """Number of failed checks (helper, not a tool)."""
    return sum(1 for r in ctx.workspace.result.quality.results if r.status == CheckStatus.FAILED)
