"""Reusable agent scripts for the bundled customer_orders.csv sample."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from tests.agents.fakes import seeded_id, submit, submit_with, task_of, tool_call, tool_evidence_ids

DEDUPE_SQL = "SELECT DISTINCT * FROM dataset"
MALICIOUS_SQL = "SELECT * FROM read_csv('/etc/passwd')"
DEDUPE_PYSPARK = "result = df.dropDuplicates()"


def investigator_script() -> list[Any]:
    def build(messages: list[BaseMessage]) -> dict[str, Any]:
        dup = seeded_id(messages, "uniqueness.duplicate_records", None)
        qty = seeded_id(messages, "validity.data_types", "quantity")
        tool_ids = tool_evidence_ids(messages)
        return {
            "summary": "Duplicates and invalid quantities found.",
            "findings": [
                {"finding_id": "F1", "title": "Exact duplicate orders", "issue_type": "duplicate", "columns": [],
                 "severity": "medium", "description": "6 exact duplicate rows.", "affected_rows": 6,
                 "evidence_ids": [dup, tool_ids[0]], "confidence_estimate": 0.9},
                {"finding_id": "F2", "title": "Non-numeric quantities", "issue_type": "invalid_type",
                 "columns": ["quantity"], "severity": "low", "description": "Text values in quantity.",
                 "affected_rows": 3, "pattern": "values shaped like 'aaa'", "evidence_ids": [qty, tool_ids[1]],
                 "confidence_estimate": 0.8},
                {"finding_id": "F3", "title": "Hallucinated issue", "issue_type": "other", "columns": ["region"],
                 "severity": "high", "description": "Made up.", "affected_rows": 999,
                 "evidence_ids": ["E-FAKE-001"], "confidence_estimate": 0.99},
            ],
            "columns_inspected": ["quantity"],
        }

    return [
        tool_call("get_check_details", {"rule_id": "uniqueness.duplicate_records", "column": None}),
        tool_call("sample_issue_rows", {"column": "quantity", "issue": "invalid_type", "limit": 5}),
        submit_with(build),
    ]


def wrong_severity_investigator_script() -> list[Any]:
    """Claims 'critical' for a check the engine rated 'medium' (grounding must correct it)."""
    def build(messages: list[BaseMessage]) -> dict[str, Any]:
        dup = seeded_id(messages, "uniqueness.duplicate_records", None)
        return {"summary": "Duplicates.", "findings": [
            {"finding_id": "F1", "title": "Duplicates", "issue_type": "duplicate", "severity": "critical",
             "description": "Exact duplicates.", "affected_rows": 6, "evidence_ids": [dup], "confidence_estimate": 0.9}]}

    return [submit_with(build)]


# Baseline for the sample: declares order_id unique and a column that no longer exists.
SAMPLE_BASELINE = (
    '{"allow_extra_columns": true, "columns": {"order_id": {"type": "numeric", "unique": true},'
    ' "legacy_code": {"type": "text", "required": true}}}'
)


def full_scripts(with_schema: bool = False) -> dict[str, list[Any]]:
    """Deterministic responses for every agent on the sample dataset."""
    scripts = {"data_investigator": investigator_script(), "root_cause_analyst": root_cause_script(),
               "remediation_agent": remediation_script()}
    if with_schema:
        scripts["schema_analyst"] = schema_script()
    return scripts


def root_cause_script(followup: bool = False) -> list[Any]:
    def build(messages: list[BaseMessage]) -> dict[str, Any]:
        findings = task_of(messages)["findings"]
        f1 = next(f for f in findings if f["finding_id"] == "F1")
        return {
            "summary": "Duplicates likely from a re-run extract.",
            "confirmed_facts": [
                {"statement": "6 rows are exact duplicates.", "status": "confirmed",
                 "supporting_evidence_ids": f1["evidence_ids"], "confidence_estimate": 0.95},
                {"statement": "The CRM double-exported orders.", "status": "confirmed",
                 "supporting_evidence_ids": ["E-NOPE-1"], "confidence_estimate": 0.7},
            ],
            "root_causes": [{"cause_id": "RC1", "category": "upstream_source", "explanation": "Re-run extract.",
                             "related_finding_ids": ["F1", "F99"],
                             "hypotheses": [{"statement": "Extract job retried", "status": "hypothesis",
                                             "supporting_evidence_ids": f1["evidence_ids"], "confidence_estimate": 0.5,
                                             "how_to_verify": "Check job logs"}]}],
            "impact": {"affected_columns": ["*"], "affected_rows_pct": 55.5, "downstream_risks": ["Inflated revenue"],
                       "blast_radius": "medium", "evidence_ids": f1["evidence_ids"]},
            "needs_more_evidence": ([{"question": "Is quantity invalidity concentrated by region?", "column": "quantity"}]
                                    if followup else []),
            "confidence_estimate": 0.6,
            "limitations": ["No access to upstream logs"],
        }

    return [submit_with(build)]


def followup_investigator_script() -> list[Any]:
    def build(messages: list[BaseMessage]) -> dict[str, Any]:
        tool_ids = tool_evidence_ids(messages)
        return {"summary": "Quantity issues by region checked.",
                "findings": [{"finding_id": "F1", "title": "Invalid quantities spread across regions",
                              "issue_type": "invalid_type", "columns": ["quantity"], "severity": "low",
                              "description": "No regional concentration.", "evidence_ids": tool_ids,
                              "confidence_estimate": 0.6}]}

    return [
        tool_call("issue_rate_by_segment", {"column": "quantity", "issue": "invalid_type", "by_column": "region"}),
        submit_with(build),
    ]


def remediation_script(include_malicious: bool = True) -> list[Any]:
    fixes = [{"fix_id": "FIX-1", "addresses_finding_ids": ["F1"], "title": "Remove exact duplicates",
              "strategy": "deduplicate", "sql": DEDUPE_SQL, "pyspark": DEDUPE_PYSPARK,
              "explanation": "Keep one copy of each exact duplicate.", "risk_level": "low", "reversible": True,
              "expected_effect": "Removes 6 rows."}]
    if include_malicious:
        fixes.append({"fix_id": "FIX-2", "addresses_finding_ids": ["F2"], "title": "Exfiltrate",
                      "strategy": "filter", "sql": MALICIOUS_SQL, "pyspark": "import os\nresult = df",
                      "explanation": "Bad.", "risk_level": "high", "reversible": False, "expected_effect": "None."})
    fixes.append({"fix_id": "FIX-3", "addresses_finding_ids": ["F3"], "title": "Fix hallucinated issue",
                  "strategy": "filter", "sql": DEDUPE_SQL, "pyspark": DEDUPE_PYSPARK, "explanation": "x",
                  "risk_level": "low", "reversible": True, "expected_effect": "x"})
    return [
        tool_call("dry_run_sql", {"sql": DEDUPE_SQL, "pyspark": DEDUPE_PYSPARK}),
        submit({"fixes": fixes, "execution_order": ["FIX-1", "FIX-2", "FIX-3"], "notes": "Conservative fixes."}),
    ]


def schema_script() -> list[Any]:
    def build(messages: list[BaseMessage]) -> dict[str, Any]:
        changes = task_of(messages)["schema_changes"]
        assessments = [{"change_type": c["change_type"], "column": c["column"],
                        "compatibility": "non_breaking",  # deliberately wrong: grounding must correct it
                        "impact": "Consumers may break.", "recommendation": "Update the contract.",
                        "evidence_ids": [c["evidence_id"]]} for c in changes]
        assessments.append({"change_type": "removed", "column": "imaginary_column", "compatibility": "breaking",
                            "impact": "x", "recommendation": "x", "evidence_ids": ["E-S-999"]})
        return {"summary": "Drift found.", "drift_detected": False, "assessments": assessments,
                "confidence_estimate": 0.7}

    return [tool_call("get_schema_diff", {}), submit_with(build)]
