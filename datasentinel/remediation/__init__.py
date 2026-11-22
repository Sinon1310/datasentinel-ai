"""Deterministic remediation safety: SQL guard, DuckDB sandbox, PySpark static checks, approval-gated execution."""

from datasentinel.remediation.audit import AuditLog
from datasentinel.remediation.executor import ApprovedFix, ExecutionOutcome, FixExecutionRecord, execute_approved_fixes
from datasentinel.remediation.pyspark_check import PySparkCheck, check_pyspark
from datasentinel.remediation.sandbox import DuckDBSandbox, SandboxError
from datasentinel.remediation.sql_guard import SqlValidation, sql_sha256, validate_sql

__all__ = [
    "ApprovedFix",
    "AuditLog",
    "DuckDBSandbox",
    "ExecutionOutcome",
    "FixExecutionRecord",
    "PySparkCheck",
    "SandboxError",
    "SqlValidation",
    "check_pyspark",
    "execute_approved_fixes",
    "sql_sha256",
    "validate_sql",
]
