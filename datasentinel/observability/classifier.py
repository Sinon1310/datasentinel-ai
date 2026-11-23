"""Deterministic, versioned failure classification.

Rules are ordered; the first match wins and its rule id and the classifier version are stored with the
failure:

1. ``reported``            a category supplied explicitly by the caller (``category_source='reported'``)
2. ``type.quality_gate``   :class:`~datasentinel.exceptions.QualityGateError`            → DATA_QUALITY
3. ``type.schema``         :class:`~datasentinel.exceptions.SchemaContractError`         → SCHEMA
4. ``type.ingestion``      :class:`~datasentinel.exceptions.IngestionError` (+ subclasses) → INGESTION
5. ``type.transformation`` ``TransformationError``, ``SandboxError``                       → TRANSFORMATION
6. ``type.resource``       ``MemoryError``, ``TimeoutError``, ``RecursionError``, ``OSError`` family → RESOURCE_RUNTIME
7. ``external_type``       third-party exception types matched by qualified *name* (never imported)
8. ``message.<name>``      patterns over the *redacted* message
9. ``fallback.unknown``    anything else                                                   → UNKNOWN

Type rules (2–7) are checked against the class hierarchy of the raised exception first, then of each
link of its cause/context chain (up to five links), so a wrapper such as ``RuntimeError`` raised
``from`` a ``QualityGateError`` is still DATA_QUALITY, while an outer typed error wins over its cause.
Message patterns (8) are weaker evidence and only apply when no type rule matched on any link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, List, Mapping, Optional, Pattern, Sequence, Tuple, Type

from datasentinel.exceptions import (
    IngestionError, QualityGateError, SchemaContractError, TransformationError,
)
from datasentinel.observability.models import CategorySource, FailureCategory
from datasentinel.observability.redaction import MAX_CHAIN_LINKS, Redactor, _qualified_name

CLASSIFIER_VERSION = "v1"

_F = FailureCategory
# (rule id, classes, engine classes matched by qualified name to avoid importing heavy modules, category)
_TYPE_RULES: Tuple[Tuple[str, Tuple[Type[BaseException], ...], FrozenSet[str], FailureCategory], ...] = (
    ("type.quality_gate", (QualityGateError,), frozenset(), _F.DATA_QUALITY),
    ("type.schema", (SchemaContractError,), frozenset(), _F.SCHEMA),
    ("type.ingestion", (IngestionError,), frozenset(), _F.INGESTION),
    ("type.transformation", (TransformationError,), frozenset({"datasentinel.remediation.sandbox.SandboxError"}),
     _F.TRANSFORMATION),
    ("type.resource", (MemoryError, TimeoutError, RecursionError, OSError), frozenset(), _F.RESOURCE_RUNTIME),
)

DEFAULT_EXTERNAL_TYPES: Mapping[str, FailureCategory] = {
    "pyarrow.lib.ArrowInvalid": _F.TRANSFORMATION,
    "pyarrow.lib.ArrowTypeError": _F.TRANSFORMATION,
    "pandas.errors.ParserError": _F.INGESTION,
    "pandas.errors.EmptyDataError": _F.INGESTION,
    "duckdb.duckdb.ConversionException": _F.TRANSFORMATION,
    "duckdb.duckdb.OutOfMemoryException": _F.RESOURCE_RUNTIME,
}

DEFAULT_MESSAGE_PATTERNS: Tuple[Tuple[str, str, FailureCategory], ...] = (
    ("out_of_memory", r"\bout of memory\b|\bOOM\b|cannot allocate memory", _F.RESOURCE_RUNTIME),
    ("disk_full", r"no space left on device|disk (is )?full|quota exceeded", _F.RESOURCE_RUNTIME),
    ("timed_out", r"\btimed? ?out\b|deadline exceeded", _F.RESOURCE_RUNTIME),
    ("schema_mismatch", r"schema mismatch|missing required column|unexpected column", _F.SCHEMA),
)

_RULE_NAME = re.compile(r"^[a-z0-9_]{1,40}$")


@dataclass(frozen=True)
class Classification:
    """The outcome of classifying one failure."""

    category: FailureCategory
    category_source: CategorySource
    rule: str
    version: str
    matched_link: Optional[int] = None  # 0 = the raised exception, n = n-th cause/context link
    matched_type: Optional[str] = None

    def as_context(self) -> dict:
        """Classification details for the failure's context (no user data)."""
        return {"rule": self.rule, "version": self.version, "matched_link": self.matched_link,
                "matched_type": self.matched_type}


@dataclass(frozen=True)
class ClassifierConfig:
    """Configurable rules 7 and 8. Names are matched as strings, so libraries are never imported."""

    external_types: Mapping[str, FailureCategory] = field(default_factory=lambda: dict(DEFAULT_EXTERNAL_TYPES))
    message_patterns: Sequence[Tuple[str, str, FailureCategory]] = DEFAULT_MESSAGE_PATTERNS

    def __post_init__(self) -> None:
        for name, category in self.external_types.items():
            if not isinstance(name, str) or not name or not isinstance(category, FailureCategory):
                raise ValueError("external_types maps qualified class names to FailureCategory values")
        for name, pattern, category in self.message_patterns:
            if not _RULE_NAME.fullmatch(name) or not isinstance(category, FailureCategory):
                raise ValueError("message pattern names are lowercase slugs; categories are FailureCategory values")
            re.compile(pattern)


def exception_chain(exc: BaseException, limit: int = MAX_CHAIN_LINKS) -> List[BaseException]:
    """The raised exception followed by up to ``limit`` cause/context links (cycle-safe)."""
    links, seen = [exc], {id(exc)}
    current = exc
    while len(links) <= limit:
        nxt = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
        if nxt is None or id(nxt) in seen:
            break
        links.append(nxt)
        seen.add(id(nxt))
        current = nxt
    return links


class FailureClassifier:
    """Maps exceptions to failure categories with ordered, versioned rules."""

    version = CLASSIFIER_VERSION

    def __init__(self, config: Optional[ClassifierConfig] = None, redactor: Optional[Redactor] = None) -> None:
        self._config = config or ClassifierConfig()
        self._redactor = redactor or Redactor()
        self._patterns: List[Tuple[str, Pattern[str], FailureCategory]] = [
            (name, re.compile(pattern, re.IGNORECASE), category)
            for name, pattern, category in self._config.message_patterns]

    def classify(self, exc: Optional[BaseException], *,
                 reported_category: Optional[FailureCategory] = None) -> Classification:
        """Classify ``exc`` (or only the reported category when there is no exception)."""
        if reported_category is not None:
            return Classification(FailureCategory(reported_category), CategorySource.REPORTED, "reported",
                                  self.version)
        if exc is None:
            return self._unknown()
        links = exception_chain(exc)
        for index, link in enumerate(links):
            match = self._match_type(link)
            if match is not None:
                rule, category, matched = match
                return Classification(category, CategorySource.CLASSIFIER, rule, self.version, index, matched)
        for index, link in enumerate(links):
            message = self._redactor.exception_message(link)
            for name, pattern, category in self._patterns:
                if pattern.search(message):
                    return Classification(category, CategorySource.CLASSIFIER, f"message.{name}", self.version,
                                          index, _qualified_name(type(link)))
        return self._unknown()

    def _match_type(self, exc: BaseException) -> Optional[Tuple[str, FailureCategory, str]]:
        mro_names = [f"{cls.__module__}.{cls.__qualname__}" for cls in type(exc).__mro__]
        for rule, types, names, category in _TYPE_RULES:
            if isinstance(exc, types) or names.intersection(mro_names):
                return rule, category, _qualified_name(type(exc))
        for cls, name in zip(type(exc).__mro__, mro_names):
            if name in self._config.external_types:
                return "external_type", self._config.external_types[name], _qualified_name(cls)
        return None

    def _unknown(self) -> Classification:
        return Classification(FailureCategory.UNKNOWN, CategorySource.CLASSIFIER, "fallback.unknown", self.version)
