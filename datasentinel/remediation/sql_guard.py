"""SQL safety validation with sqlglot.

Policy (allow-list): exactly one read-only query (``SELECT`` / ``WITH … SELECT`` / set
operations) that reads only the relation ``dataset`` or CTEs it defines. Everything
else is rejected:

* DDL / DML / ``COPY`` / ``ATTACH`` / ``INSTALL`` / ``LOAD`` / ``PRAGMA`` / ``SET`` / commands
* ``SELECT … INTO``
* schema- or catalog-qualified tables, table functions (``FROM read_csv(...)``)
* file, network, environment and introspection functions (``read_*``, ``*_scan``, ``glob``, ``getenv`` …)

This guard is one of two independent layers; the DuckDB sandbox additionally runs with
external access disabled. The SHA-256 of the *exact* SQL text binds human approvals.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

ALLOWED_RELATION = "dataset"
MAX_SQL_LENGTH = 10_000

_DENIED_FUNCTION = re.compile(
    r"^(read_\w*|\w*_scan|glob|getenv|load_extension|sniff_csv|duckdb_\w*|pragma_\w*|current_setting|"
    r"query|query_table|system|shell|httpfs\w*|install\w*|load|export_database|import_database)$",
    re.IGNORECASE,
)
_DENIED_NODE_NAMES = (
    "DDL", "DML", "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
    "Command", "Copy", "Pragma", "Set", "Use", "Attach", "Detach", "Install", "Load", "Transaction",
    "Commit", "Rollback", "Grant", "Describe", "Into", "Export",
)
_DENIED_NODES = tuple(getattr(exp, name) for name in _DENIED_NODE_NAMES if hasattr(exp, name))


def sql_sha256(sql: str) -> str:
    """SHA-256 of the exact SQL text (no normalisation — any edit changes the hash)."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


@dataclass
class SqlValidation:
    """Outcome of validating a SQL statement."""

    ok: bool
    sha256: str
    violations: List[str] = field(default_factory=list)
    statement_type: Optional[str] = None


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name)
    return node.sql_name()


def validate_sql(sql: str) -> SqlValidation:
    """Validate ``sql`` against the read-only remediation policy."""
    digest = sql_sha256(sql if isinstance(sql, str) else "")
    if not isinstance(sql, str) or not sql.strip():
        return SqlValidation(False, digest, ["SQL is empty."])
    if len(sql) > MAX_SQL_LENGTH:
        return SqlValidation(False, digest, [f"SQL exceeds {MAX_SQL_LENGTH} characters."])
    try:
        trees = [t for t in sqlglot.parse(sql, read="duckdb") if t is not None]
    except (ParseError, TokenError) as exc:
        return SqlValidation(False, digest, [f"SQL syntax error: {str(exc).splitlines()[0][:200]}"])

    violations: List[str] = []
    if len(trees) != 1:
        return SqlValidation(False, digest, [f"Exactly one statement is allowed; found {len(trees)}."])
    tree = trees[0]
    statement_type = type(tree).__name__
    if not isinstance(tree, exp.Query):
        violations.append(f"Only read-only SELECT queries are allowed; got {statement_type}.")

    for node in tree.walk():
        if isinstance(node, _DENIED_NODES):
            violations.append(f"Forbidden construct: {type(node).__name__}.")
    if tree.args.get("into") is not None:
        violations.append("SELECT ... INTO is not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    if not any(t.name.lower() == ALLOWED_RELATION for t in tree.find_all(exp.Table)):
        violations.append(f"The query must read from the '{ALLOWED_RELATION}' relation.")
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            violations.append(f"Table functions are not allowed: {_function_name(table.this)}().")
            continue
        if table.args.get("db") or table.args.get("catalog"):
            violations.append(f"Qualified table names are not allowed: {table.sql(dialect='duckdb')}.")
        name = table.name.lower()
        if name and name != ALLOWED_RELATION and name not in cte_names:
            violations.append(f"Only the '{ALLOWED_RELATION}' relation (and CTEs) may be read; found '{table.name}'.")

    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if _DENIED_FUNCTION.match(name or ""):
            violations.append(f"Forbidden function: {name}().")

    unique = list(dict.fromkeys(violations))
    return SqlValidation(not unique, digest, unique, statement_type)
