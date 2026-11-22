"""Security-focused tests for SQL validation, the DuckDB sandbox, PySpark checks and execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datasentinel.remediation import (
    ApprovedFix,
    AuditLog,
    DuckDBSandbox,
    SandboxError,
    check_pyspark,
    execute_approved_fixes,
    sql_sha256,
    validate_sql,
)

SAFE_SQL = [
    "SELECT * FROM dataset",
    "SELECT DISTINCT * FROM dataset",
    "WITH d AS (SELECT * FROM dataset) SELECT * FROM d",
    "SELECT * FROM dataset UNION ALL SELECT * FROM dataset",
    "SELECT a, COALESCE(b, 0) AS b FROM dataset WHERE a IS NOT NULL",
    "SELECT * FROM dataset -- trailing comment",
]

MALICIOUS_SQL = [
    "SELECT * FROM dataset; DROP TABLE dataset",
    "DROP TABLE dataset",
    "DELETE FROM dataset",
    "UPDATE dataset SET a = 1",
    "INSERT INTO dataset VALUES (1)",
    "CREATE TABLE x AS SELECT * FROM dataset",
    "ATTACH 'other.db'",
    "COPY dataset TO '/tmp/leak.csv'",
    "INSTALL httpfs",
    "LOAD httpfs",
    "PRAGMA database_list",
    "SET enable_external_access = true",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://bucket/x')",
    "SELECT * FROM 'secrets.csv'",
    "SELECT * FROM glob('*')",
    "SELECT getenv('GROQ_API_KEY')",
    "SELECT * FROM duckdb_settings()",
    "SELECT * FROM main.dataset",
    "SELECT * INTO stolen FROM dataset",
    "SELECT * FROM other_table",
    "SELECT 1",
    "SELECT * FR",
    "",
    "SELECT * FROM dataset WHERE a = '" + "x" * 10_001 + "'",
]


class TestSqlGuard:
    @pytest.mark.parametrize("sql", SAFE_SQL)
    def test_allows_read_only_queries(self, sql: str) -> None:
        result = validate_sql(sql)
        assert result.ok, result.violations

    @pytest.mark.parametrize("sql", MALICIOUS_SQL)
    def test_blocks_dangerous_queries(self, sql: str) -> None:
        result = validate_sql(sql)
        assert not result.ok and result.violations

    def test_hash_binds_exact_text(self) -> None:
        assert validate_sql("SELECT * FROM dataset").sha256 == sql_sha256("SELECT * FROM dataset")
        assert sql_sha256("SELECT * FROM dataset") != sql_sha256("SELECT *  FROM dataset")


class TestSandbox:
    def test_runs_query_without_mutating_input(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        before = df.copy()
        out = DuckDBSandbox().run("SELECT DISTINCT * FROM dataset ORDER BY a", df)
        assert len(out) == 2
        pd.testing.assert_frame_equal(df, before)

    def test_rejects_policy_violations_before_execution(self) -> None:
        with pytest.raises(SandboxError, match="policy"):
            DuckDBSandbox().run("SELECT * FROM read_csv('/etc/passwd')", pd.DataFrame({"a": [1]}))

    def test_external_access_disabled_at_engine_level(self) -> None:
        # Defence in depth: even bypassing the guard, DuckDB itself refuses file access.
        sandbox = DuckDBSandbox()
        connection = sandbox._connect()
        try:
            with pytest.raises(Exception):
                connection.execute("SELECT * FROM read_csv('/etc/passwd')")
            with pytest.raises(Exception):
                connection.execute("SET enable_external_access = true")
        finally:
            connection.close()

    def test_runtime_errors_become_sandbox_errors(self) -> None:
        with pytest.raises(SandboxError, match="failed"):
            DuckDBSandbox().run("SELECT missing_column FROM dataset", pd.DataFrame({"a": [1]}))

    def test_dry_run_row_cap(self) -> None:
        out = DuckDBSandbox().run("SELECT * FROM dataset", pd.DataFrame({"a": range(100)}), max_input_rows=10)
        assert len(out) == 10


class TestPySparkCheck:
    def test_accepts_pure_transformation(self) -> None:
        code = (
            "from pyspark.sql import functions as F\n"
            "result = df.dropDuplicates().withColumn('amount', F.when(F.col('amount') > 1000, None)"
            ".otherwise(F.col('amount')))"
        )
        assert check_pyspark(code).ok

    @pytest.mark.parametrize("code, expected", [
        ("import os\nresult = df", "Forbidden import"),
        ("result = df\ndf.write.saveAsTable('x')", "I/O"),
        ("result = spark.sql('DROP TABLE x')", "spark"),
        ("result = eval('df')", "eval"),
        ("result = df.__class__", "dunder"),
        ("def f():\n    pass\nresult = df", "FunctionDef"),
        ("for i in range(3):\n    pass\nresult = df", "For"),
        ("x = df.collect()\nresult = df", "collect"),
        ("result = df.rdd.map(lambda r: r)", "rdd"),
        ("out = df.dropna()", "result"),
        ("result = other.dropna()", "'df'"),
        ("result = df.", "syntax"),
    ])
    def test_rejects_unsafe_code(self, code: str, expected: str) -> None:
        check = check_pyspark(code)
        assert not check.ok
        assert any(expected.lower() in v.lower() for v in check.violations), check.violations


class TestExecutor:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2, 2, 3], "amount": [10.0, 20.0, 20.0, 99999.0]})

    def _fix(self, fix_id: str, sql: str, approved_sql: str = None) -> ApprovedFix:
        return ApprovedFix(fix_id, sql, sql_sha256(approved_sql if approved_sql is not None else sql), "tester")

    def test_executes_approved_fixes_in_order_on_a_copy(self, df: pd.DataFrame, tmp_path: Path) -> None:
        original = df.copy()
        audit = AuditLog(tmp_path / "audit.jsonl")
        outcome = execute_approved_fixes(df, [
            self._fix("F1", "SELECT DISTINCT * FROM dataset"),
            self._fix("F2", "SELECT id, CASE WHEN amount > 1000 THEN NULL ELSE amount END AS amount FROM dataset"),
        ], audit=audit)
        assert outcome.executed_ids == ["F1", "F2"]
        assert len(outcome.output) == 3 and outcome.output["amount"].isna().sum() == 1
        pd.testing.assert_frame_equal(df, original)  # original never modified
        entries = audit.read()
        assert [e["status"] for e in entries] == ["executed", "executed"]
        assert all(len(e["sql_sha256"]) == 64 for e in entries)

    def test_blocks_tampered_sql(self, df: pd.DataFrame) -> None:
        tampered = self._fix("F1", "SELECT * FROM dataset WHERE id > 1", approved_sql="SELECT * FROM dataset")
        outcome = execute_approved_fixes(df, [tampered])
        assert outcome.records[0].status == "blocked" and "hash" in outcome.records[0].reason
        assert outcome.output is None

    def test_revalidates_policy_even_with_matching_hash(self, df: pd.DataFrame) -> None:
        outcome = execute_approved_fixes(df, [self._fix("F1", "SELECT * FROM read_csv('/etc/passwd')")])
        assert outcome.records[0].status == "blocked" and "Policy" in outcome.records[0].reason

    def test_blocks_column_changes_and_records_failures(self, df: pd.DataFrame) -> None:
        outcome = execute_approved_fixes(df, [
            self._fix("F1", "SELECT id FROM dataset"),
            self._fix("F2", "SELECT nope FROM dataset"),
        ])
        assert [r.status for r in outcome.records] == ["blocked", "failed"]
        assert outcome.output is None


def test_audit_log_is_append_only(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")
    log.record("approval", fix_id="F1")
    AuditLog(tmp_path / "a.jsonl").record("approval", fix_id="F2")
    assert [e["fix_id"] for e in log.read()] == ["F1", "F2"]
