"""Static validation of generated PySpark code. Generated code is NEVER executed.

The code must be a pure DataFrame transformation of an input ``df`` that assigns its
output to ``result``. The AST is checked against an allow-list policy:

* imports only from ``pyspark.sql`` (functions, types, window)
* no I/O or session access (``read``, ``write``, ``save*``, ``spark.sql``, ``collect``, ``rdd`` …)
* no dangerous builtins (``exec``, ``eval``, ``open``, ``__import__``, ``getattr`` …)
* no dunder / private attribute access, no class/function definitions, loops, ``with``, ``try``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

MAX_CODE_LENGTH = 10_000
_ALLOWED_MODULES = {"pyspark.sql", "pyspark.sql.functions", "pyspark.sql.types", "pyspark.sql.window"}
_DENIED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "memoryview", "os", "sys", "subprocess", "spark", "sc",
}
_DENIED_ATTRIBUTES = {
    "read", "readStream", "write", "writeStream", "save", "saveAsTable", "insertInto", "jdbc", "csv", "parquet",
    "json", "orc", "text", "load", "sql", "collect", "toPandas", "toLocalIterator", "foreach", "foreachPartition",
    "rdd", "sparkContext", "sparkSession", "createOrReplaceTempView", "createTempView", "createGlobalTempView",
    "createOrReplaceGlobalTempView", "registerTempTable", "udf", "pandas_udf", "mapInPandas", "applyInPandas",
}
_DENIED_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith, ast.While, ast.For,
    ast.AsyncFor, ast.Try, ast.Global, ast.Nonlocal, ast.Delete, ast.Raise, ast.Await, ast.Yield, ast.YieldFrom,
)


@dataclass
class PySparkCheck:
    """Outcome of static PySpark validation."""

    ok: bool
    violations: List[str] = field(default_factory=list)


def check_pyspark(code: str) -> PySparkCheck:
    """Statically validate generated PySpark code without executing it."""
    if not isinstance(code, str) or not code.strip():
        return PySparkCheck(False, ["PySpark code is empty."])
    if len(code) > MAX_CODE_LENGTH:
        return PySparkCheck(False, [f"PySpark code exceeds {MAX_CODE_LENGTH} characters."])
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PySparkCheck(False, [f"Python syntax error on line {exc.lineno}: {exc.msg}"])

    violations: List[str] = []
    assigns_result = False
    uses_df = False
    for node in ast.walk(tree):
        if isinstance(node, _DENIED_NODES):
            violations.append(f"Forbidden construct: {type(node).__name__} (line {getattr(node, 'lineno', '?')}).")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_MODULES:
                    violations.append(f"Forbidden import: {alias.name}.")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_MODULES:
                violations.append(f"Forbidden import from: {node.module}.")
        elif isinstance(node, ast.Name):
            if node.id in _DENIED_NAMES:
                violations.append(f"Forbidden name: {node.id}.")
            if node.id == "df" and isinstance(node.ctx, ast.Load):
                uses_df = True
            if node.id == "result" and isinstance(node.ctx, ast.Store):
                assigns_result = True
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"Private or dunder attribute access: .{node.attr}.")
            elif node.attr in _DENIED_ATTRIBUTES:
                violations.append(f"Forbidden I/O or session operation: .{node.attr}.")
    if not uses_df:
        violations.append("Code must transform the input DataFrame named 'df'.")
    if not assigns_result:
        violations.append("Code must assign its output DataFrame to 'result'.")
    unique = list(dict.fromkeys(violations))
    return PySparkCheck(not unique, unique)
