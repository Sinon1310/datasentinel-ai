"""Deterministic schema drift detection.

Compares a *baseline* (an uploaded :class:`SchemaContract`, or a previously exported
``SchemaReport`` JSON) with the schema inferred for the current dataset, and classifies
every difference as ``breaking``, ``non_breaking`` or ``needs_review``.

The classification rules are deterministic so the Schema Analyst agent *interprets*
drift rather than inventing it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from datasentinel.exceptions import SchemaContractError
from datasentinel.schema.contract import ColumnContract, SchemaContract
from datasentinel.schema.models import SchemaReport, SemanticType
from datasentinel.utils.serialization import to_builtin

ChangeType = Literal[
    "added", "removed", "type_changed", "nullability_changed", "uniqueness_changed", "rename_candidate"
]
Compatibility = Literal["breaking", "non_breaking", "needs_review"]

_STRING_FAMILY = {SemanticType.CATEGORICAL, SemanticType.TEXT}
_RENAME_NAME_SIMILARITY = 0.6


@dataclass
class SchemaChangeRecord:
    """One deterministic difference between baseline and current schema."""

    change_type: ChangeType
    column: str
    before: Optional[str]
    after: Optional[str]
    compatibility: Compatibility
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaDiff:
    """Result of comparing a baseline with the current schema."""

    baseline_source: str
    baseline_columns: int
    current_columns: int
    changes: List[SchemaChangeRecord] = field(default_factory=list)

    @property
    def drift_detected(self) -> bool:
        """True when any difference exists."""
        return bool(self.changes)

    def count(self, compatibility: Compatibility) -> int:
        """Number of changes with the given compatibility class."""
        return sum(1 for c in self.changes if c.compatibility == compatibility)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe representation."""
        data = to_builtin(self)
        data["drift_detected"] = self.drift_detected
        data["summary"] = {k: self.count(k) for k in ("breaking", "non_breaking", "needs_review")}
        return data


def load_baseline(text: str) -> tuple[SchemaContract, str]:
    """Parse a baseline from either supported format.

    Supported:
      * a schema contract: ``{"columns": {"name": {...}}, ...}``
      * an exported schema report: ``{"columns": [{"name": ..., "semantic_type": ...}], ...}``

    Returns:
        ``(contract, source)`` where ``source`` is ``"contract"`` or ``"schema_report"``.

    Raises:
        SchemaContractError: If the text is not a recognised baseline.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaContractError(f"Baseline is not valid JSON: {exc.msg} (line {exc.lineno}).") from exc
    if not isinstance(data, dict):
        raise SchemaContractError("Baseline must be a JSON object.")
    columns = data.get("columns")
    if isinstance(columns, dict):
        return SchemaContract.from_dict(data), "contract"
    if isinstance(columns, list):
        return _contract_from_schema_export(data), "schema_report"
    raise SchemaContractError("Baseline must be a schema contract or an exported schema report JSON.")


def _contract_from_schema_export(data: Dict[str, Any]) -> SchemaContract:
    valid_types = {t.value for t in SemanticType}
    keys = {
        pk["columns"][0]
        for pk in data.get("primary_key_candidates", []) or []
        if isinstance(pk, dict) and isinstance(pk.get("columns"), list) and len(pk["columns"]) == 1
    }
    contracts: Dict[str, ColumnContract] = {}
    for item in data["columns"]:
        if not isinstance(item, dict) or "name" not in item or item.get("semantic_type") not in valid_types:
            raise SchemaContractError("Exported schema JSON has a column without a valid 'name'/'semantic_type'.")
        stype = SemanticType(item["semantic_type"])
        name = str(item["name"])
        contracts[name] = ColumnContract(
            name=name,
            type=None if stype == SemanticType.EMPTY else stype,
            required=True,
            nullable=bool(item.get("nullable", True)),
            unique=name in keys,
        )
    if not contracts:
        raise SchemaContractError("Exported schema JSON contains no columns.")
    return SchemaContract(columns=contracts, allow_extra_columns=True)


def _type_compatibility(before: SemanticType, after: SemanticType) -> tuple[Compatibility, str]:
    if before in _STRING_FAMILY and after in _STRING_FAMILY:
        return "non_breaking", "categorical and text are both string-typed"
    if after in _STRING_FAMILY:
        return "breaking", f"{before.value} values now arrive as strings; typed consumers will fail or mis-sort"
    if after == SemanticType.EMPTY:
        return "needs_review", "column is now entirely empty"
    return "breaking", f"semantic type changed from {before.value} to {after.value}"


def compute_schema_diff(
    baseline: SchemaContract,
    current: SchemaReport,
    raw: pd.DataFrame,
    baseline_source: str = "contract",
) -> SchemaDiff:
    """Compare ``baseline`` with ``current`` and classify every difference."""
    actual = {c.name: c for c in current.columns}
    diff = SchemaDiff(baseline_source=baseline_source, baseline_columns=len(baseline.columns),
                      current_columns=len(actual))
    removed: List[str] = []

    for name, expected in baseline.columns.items():
        column = actual.get(name)
        if column is None:
            removed.append(name)
            diff.changes.append(SchemaChangeRecord(
                "removed", name, expected.type.value if expected.type else "any", None,
                "breaking" if expected.required else "non_breaking",
                "required column missing" if expected.required else "optional column missing",
            ))
            continue
        if expected.type is not None and column.semantic_type != expected.type:
            compat, reason = _type_compatibility(expected.type, column.semantic_type)
            diff.changes.append(SchemaChangeRecord(
                "type_changed", name, expected.type.value, column.semantic_type.value, compat, reason,
                {"parse_success_ratio": column.parse_success_ratio},
            ))
        if not expected.nullable and column.null_count > 0:
            diff.changes.append(SchemaChangeRecord(
                "nullability_changed", name, "non-nullable", "nullable", "breaking",
                f"{column.null_count} missing value(s) in a column declared non-nullable",
                {"null_count": column.null_count},
            ))
        if expected.unique:
            duplicates = int(raw[name].dropna().duplicated().sum())
            if duplicates:
                diff.changes.append(SchemaChangeRecord(
                    "uniqueness_changed", name, "unique", "non-unique", "breaking",
                    f"{duplicates} duplicate value(s) in a column declared unique",
                    {"duplicate_values": duplicates},
                ))

    added = [n for n in actual if n not in baseline.columns]
    for name in added:
        diff.changes.append(SchemaChangeRecord(
            "added", name, None, actual[name].semantic_type.value,
            "non_breaking" if baseline.allow_extra_columns else "needs_review",
            "new column" if baseline.allow_extra_columns else "new column not allowed by contract",
        ))

    for old in removed:
        for new in added:
            similarity = difflib.SequenceMatcher(None, old.lower(), new.lower()).ratio()
            expected_type = baseline.columns[old].type
            same_type = expected_type is None or expected_type == actual[new].semantic_type
            if similarity >= _RENAME_NAME_SIMILARITY and same_type:
                diff.changes.append(SchemaChangeRecord(
                    "rename_candidate", new, old, new, "needs_review",
                    f"'{old}' is missing and '{new}' is new with a similar name and compatible type",
                    {"name_similarity": round(similarity, 3)},
                ))
    return diff
