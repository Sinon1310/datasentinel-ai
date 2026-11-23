"""Failure classification: rule table, precedence, cause chains, external names, messages and fallback."""

from __future__ import annotations

import pytest

from datasentinel.exceptions import (
    DatasetParseError, EmptyDatasetError, QualityGateError, SchemaContractError, TransformationError,
)
from datasentinel.observability import CategorySource, ClassifierConfig, FailureCategory, FailureClassifier
from datasentinel.remediation.sandbox import SandboxError

C = FailureClassifier()
F = FailureCategory


def _chained(outer: BaseException, inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as exc:
            raise outer from exc
    except BaseException as wrapped:
        return wrapped


@pytest.mark.parametrize("exc,category,rule", [
    (QualityGateError("gate"), F.DATA_QUALITY, "type.quality_gate"),
    (SchemaContractError("bad contract"), F.SCHEMA, "type.schema"),
    (DatasetParseError("parse"), F.INGESTION, "type.ingestion"),
    (EmptyDatasetError("empty"), F.INGESTION, "type.ingestion"),
    (TransformationError("t"), F.TRANSFORMATION, "type.transformation"),
    (SandboxError("sql"), F.TRANSFORMATION, "type.transformation"),
    (MemoryError(), F.RESOURCE_RUNTIME, "type.resource"),
    (TimeoutError(), F.RESOURCE_RUNTIME, "type.resource"),
    (RecursionError(), F.RESOURCE_RUNTIME, "type.resource"),
    (PermissionError("denied"), F.RESOURCE_RUNTIME, "type.resource"),
    (ValueError("Out of memory while sorting"), F.RESOURCE_RUNTIME, "message.out_of_memory"),
    (RuntimeError("No space left on device"), F.RESOURCE_RUNTIME, "message.disk_full"),
    (RuntimeError("missing required column order_id"), F.SCHEMA, "message.schema_mismatch"),
    (KeyError("order_id"), F.UNKNOWN, "fallback.unknown"),
    (ValueError("bad value"), F.UNKNOWN, "fallback.unknown"),
])
def test_rule_table(exc: BaseException, category: FailureCategory, rule: str) -> None:
    result = C.classify(exc)
    assert (result.category, result.rule, result.version, result.category_source) == \
           (category, rule, "v1", CategorySource.CLASSIFIER)


def test_reported_category_wins_over_everything() -> None:
    result = C.classify(QualityGateError("x"), reported_category=F.INGESTION)
    assert (result.category, result.category_source, result.rule) == (F.INGESTION, CategorySource.REPORTED,
                                                                       "reported")
    assert C.classify(None, reported_category=F.SCHEMA).category is F.SCHEMA
    assert C.classify(None).category is F.UNKNOWN


def test_cause_chain_is_followed_and_link_recorded() -> None:
    result = C.classify(_chained(RuntimeError("step failed"), QualityGateError("gate")))
    assert (result.category, result.matched_link) == (F.DATA_QUALITY, 1)
    assert result.matched_type == "datasentinel.exceptions.QualityGateError"


def test_outer_typed_error_wins_over_its_cause() -> None:
    assert C.classify(_chained(TransformationError("t"), MemoryError())).category is F.TRANSFORMATION


def test_type_rules_beat_message_patterns_on_any_link() -> None:
    exc = _chained(RuntimeError("timed out waiting"), SchemaContractError("contract"))
    assert C.classify(exc).rule == "type.schema"


def test_external_types_match_by_name_without_imports() -> None:
    ArrowInvalid = type("ArrowInvalid", (Exception,), {"__module__": "pyarrow.lib"})
    result = C.classify(ArrowInvalid("cast failed"))
    assert (result.category, result.rule) == (F.TRANSFORMATION, "external_type")
    Sub = type("MyParserError", (ArrowInvalid,), {"__module__": "my.pkg"})
    assert C.classify(Sub("x")).category is F.TRANSFORMATION  # matched through the class hierarchy


def test_configurable_rules() -> None:
    custom = FailureClassifier(ClassifierConfig(external_types={"acme.errors.Throttled": F.RESOURCE_RUNTIME},
                                                message_patterns=(("vendor_down", r"vendor api unavailable",
                                                                   F.INGESTION),)))
    Throttled = type("Throttled", (Exception,), {"__module__": "acme.errors"})
    assert custom.classify(Throttled()).category is F.RESOURCE_RUNTIME
    assert custom.classify(RuntimeError("Vendor API unavailable")).rule == "message.vendor_down"
    assert custom.classify(ValueError("Out of memory")).category is F.UNKNOWN  # defaults replaced
    with pytest.raises(ValueError):
        ClassifierConfig(message_patterns=(("Bad Name", "x", F.UNKNOWN),))
    with pytest.raises(ValueError):
        ClassifierConfig(external_types={"x.Y": "SCHEMA"})  # type: ignore[dict-item]


def test_message_patterns_see_only_redacted_text() -> None:
    custom = FailureClassifier(ClassifierConfig(message_patterns=(("leak", r"hunter2", F.SCHEMA),)))
    assert custom.classify(RuntimeError("password=hunter2")).category is F.UNKNOWN


def test_cyclic_chains_terminate() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert C.classify(a).category is F.UNKNOWN


def test_classification_is_deterministic() -> None:
    exc = _chained(RuntimeError("x"), DatasetParseError("y"))
    assert {C.classify(exc) for _ in range(20)} == {C.classify(exc)}
