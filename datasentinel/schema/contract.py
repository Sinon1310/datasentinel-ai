"""Schema contracts: the *expected* schema a dataset is validated against.

A contract can be authored by hand as JSON, or bootstrapped from an inferred
:class:`~datasentinel.schema.models.SchemaReport` (``SchemaContract.from_schema_report``),
edited, and then reused to validate future loads of the same feed — a lightweight
data-contract workflow.

Example JSON::

    {
      "allow_extra_columns": true,
      "columns": {
        "order_id":   {"type": "numeric", "required": true, "nullable": false, "unique": true},
        "order_date": {"type": "datetime", "nullable": false}
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datasentinel.exceptions import SchemaContractError
from datasentinel.schema.models import SchemaReport, SemanticType

_ALLOWED_COLUMN_KEYS = {"type", "required", "nullable", "unique"}


@dataclass
class ColumnContract:
    """Expectations for a single column."""

    name: str
    type: Optional[SemanticType] = None
    required: bool = True
    nullable: bool = True
    unique: bool = False


@dataclass
class SchemaContract:
    """Expected schema for a dataset."""

    columns: Dict[str, ColumnContract] = field(default_factory=dict)
    allow_extra_columns: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> "SchemaContract":
        """Build and validate a contract from a parsed JSON structure.

        Raises:
            SchemaContractError: If the structure is invalid.
        """
        if not isinstance(data, dict):
            raise SchemaContractError("Schema contract must be a JSON object.")
        raw_columns = data.get("columns")
        if not isinstance(raw_columns, dict) or not raw_columns:
            raise SchemaContractError("Schema contract must define a non-empty 'columns' object.")
        allow_extra = data.get("allow_extra_columns", True)
        if not isinstance(allow_extra, bool):
            raise SchemaContractError("'allow_extra_columns' must be true or false.")

        valid_types = {t.value for t in SemanticType}
        columns: Dict[str, ColumnContract] = {}
        for name, spec in raw_columns.items():
            if spec is None:
                spec = {}
            if not isinstance(spec, dict):
                raise SchemaContractError(f"Column '{name}': specification must be an object.")
            unknown = set(spec) - _ALLOWED_COLUMN_KEYS
            if unknown:
                raise SchemaContractError(f"Column '{name}': unknown keys {sorted(unknown)}.")
            type_value = spec.get("type")
            if type_value is not None and type_value not in valid_types:
                raise SchemaContractError(
                    f"Column '{name}': invalid type '{type_value}'. Expected one of {sorted(valid_types)}."
                )
            for flag in ("required", "nullable", "unique"):
                if flag in spec and not isinstance(spec[flag], bool):
                    raise SchemaContractError(f"Column '{name}': '{flag}' must be true or false.")
            columns[str(name)] = ColumnContract(
                name=str(name),
                type=SemanticType(type_value) if type_value else None,
                required=spec.get("required", True),
                nullable=spec.get("nullable", True),
                unique=spec.get("unique", False),
            )
        return cls(columns=columns, allow_extra_columns=allow_extra)

    @classmethod
    def from_json(cls, text: str) -> "SchemaContract":
        """Parse a contract from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaContractError(f"Schema contract is not valid JSON: {exc.msg} (line {exc.lineno}).") from exc
        return cls.from_dict(data)

    @classmethod
    def from_schema_report(cls, report: SchemaReport, allow_extra_columns: bool = False) -> "SchemaContract":
        """Bootstrap a contract that the given (inferred) schema satisfies."""
        unique_columns = {pk.columns[0] for pk in report.primary_key_candidates if not pk.is_composite}
        columns = {
            c.name: ColumnContract(
                name=c.name,
                type=None if c.semantic_type == SemanticType.EMPTY else c.semantic_type,
                required=True,
                nullable=c.nullable,
                unique=c.name in unique_columns,
            )
            for c in report.columns
        }
        return cls(columns=columns, allow_extra_columns=allow_extra_columns)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the JSON contract format."""
        return {
            "allow_extra_columns": self.allow_extra_columns,
            "columns": {
                name: {
                    "type": c.type.value if c.type else None,
                    "required": c.required,
                    "nullable": c.nullable,
                    "unique": c.unique,
                }
                for name, c in self.columns.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON text."""
        return json.dumps(self.to_dict(), indent=indent)
