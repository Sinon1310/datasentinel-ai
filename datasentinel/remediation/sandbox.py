"""Isolated DuckDB execution of validated remediation SQL.

Each run uses a fresh in-memory connection with external access disabled (no files,
network, extensions), locked configuration, memory/thread limits and a wall-clock timeout.
The input DataFrame is registered as the read-only view ``dataset``; the result is a new
DataFrame. Inputs are never mutated.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import duckdb
import pandas as pd

from datasentinel.exceptions import DataSentinelError
from datasentinel.remediation.sql_guard import ALLOWED_RELATION, validate_sql

logger = logging.getLogger(__name__)


class SandboxError(DataSentinelError):
    """SQL was rejected or failed inside the sandbox."""


class DuckDBSandbox:
    """Run validated SELECT statements against a DataFrame in isolation.

    Args:
        memory_limit: DuckDB memory limit (e.g. ``"1GB"``).
        threads: Worker threads.
        timeout_seconds: Wall-clock timeout; the query is interrupted when exceeded.
    """

    def __init__(self, memory_limit: str = "1GB", threads: int = 2, timeout_seconds: float = 30.0) -> None:
        self.memory_limit = memory_limit
        self.threads = threads
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            ":memory:",
            config={
                "enable_external_access": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
                "memory_limit": self.memory_limit,
                "threads": self.threads,
                "lock_configuration": True,
            },
        )

    def run(self, sql: str, df: pd.DataFrame, max_input_rows: Optional[int] = None) -> pd.DataFrame:
        """Validate and execute ``sql`` over ``df`` and return the result.

        Args:
            sql: A query reading from ``dataset``.
            df: Input data (not modified).
            max_input_rows: If set, only the first N rows are exposed (dry runs).

        Raises:
            SandboxError: If validation fails, the query errors, or it times out.
        """
        validation = validate_sql(sql)
        if not validation.ok:
            raise SandboxError("SQL rejected by policy: " + "; ".join(validation.violations))
        source = df.head(max_input_rows) if max_input_rows else df
        connection = self._connect()
        timed_out = threading.Event()

        def _interrupt() -> None:
            timed_out.set()
            connection.interrupt()

        timer = threading.Timer(self.timeout_seconds, _interrupt)
        try:
            connection.register(ALLOWED_RELATION, source)
            timer.start()
            result = connection.execute(sql).fetch_df()
        except duckdb.Error as exc:
            if timed_out.is_set():
                raise SandboxError(f"Query exceeded the {self.timeout_seconds:.0f}s timeout.") from exc
            raise SandboxError(f"Query failed: {str(exc).splitlines()[0][:300]}") from exc
        finally:
            timer.cancel()
            connection.close()
        logger.info("Sandbox query returned %d rows x %d columns", result.shape[0], result.shape[1])
        return result
