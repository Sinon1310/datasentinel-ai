"""Pydantic models exchanged between agents, deterministic nodes and the UI.

Two families:

* **Agent submissions** (``InvestigatorReport``, ``SchemaAnalysisReport``, ``RootCauseReport``,
  ``RemediationPlan``) — produced by LLMs, validated by Pydantic, then *grounded* by
  deterministic code. Fields marked "system" are overwritten during grounding.
* **System records** (``Evidence``, ``FixValidation``, ``ApprovalDecision``, ``ExecutionResult``,
  ``VerificationResult``, ``QualityDigest``, ``InvestigationReport``) — produced only by code.

Confidence values are LLM *estimates*, not calibrated probabilities (requirement C).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

IssueType = Literal["missing", "duplicate", "invalid_type", "invalid_date", "outlier", "schema", "other"]
SeverityLevel = Literal["low", "medium", "high", "critical"]
ClaimStatus = Literal["confirmed", "hypothesis"]
ChangeType = Literal["added", "removed", "type_changed", "nullability_changed", "uniqueness_changed", "rename_candidate"]
Compatibility = Literal["breaking", "non_breaking", "needs_review"]

_CONFIDENCE = "Your uncalibrated estimate in [0, 1]. It is NOT a probability; explain limitations separately."


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


SUBMISSION_CONTEXT = {"submission": True}
"""Validation context used for LLM submissions; enables submission-only rules."""


def _require_citation(value: list[str], info: ValidationInfo) -> list[str]:
    # Citations are mandatory when an LLM submits a claim. After grounding removes invalid ids, a claim may
    # legitimately hold zero citations (it is then a hypothesis), so stored records must remain valid.
    if info.context and info.context.get("submission") and not value:
        raise ValueError("cite at least one evidence id")
    return value


# ---------------------------------------------------------------------- evidence


class Evidence(_Model):
    """An immutable, citable record of deterministic output (system)."""

    evidence_id: str
    source: Literal["quality_engine", "drift_engine", "tool"]
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    produced_by: str
    created_at: str


# ---------------------------------------------------------------------- investigator


class Finding(_Model):
    """An investigated data quality issue."""

    finding_id: str = Field(description="Short unique id, e.g. 'F1'.", max_length=20)
    title: str = Field(max_length=200)
    issue_type: IssueType
    columns: list[str] = Field(default_factory=list, max_length=20)
    severity: SeverityLevel
    status: ClaimStatus = Field(default="confirmed", description="'confirmed' only if directly shown by cited evidence.")
    description: str = Field(max_length=2000)
    pattern: str | None = Field(default=None, max_length=500, description="Where/when the issue concentrates, if observed.")
    affected_rows: int | None = Field(default=None, ge=0, description="Must equal a number in the cited evidence.")
    affected_pct: float | None = Field(default=None, ge=0, le=100, description="Must equal a number in the cited evidence.")
    evidence_ids: list[str] = Field(default_factory=list, max_length=20,
                                    description="Ids of evidence supporting this finding (at least one).")
    confidence_estimate: float = Field(ge=0, le=1, description=_CONFIDENCE)
    limitations: list[str] = Field(default_factory=list, max_length=10)
    grounding_notes: list[str] = Field(default_factory=list, description="System field; leave empty.")

    _cite = field_validator("evidence_ids")(_require_citation)


class InvestigatorReport(_Model):
    """Data Investigator output."""

    summary: str = Field(max_length=3000)
    findings: list[Finding] = Field(default_factory=list, max_length=30)
    columns_inspected: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------- schema analyst


class SchemaChangeAssessment(_Model):
    """Interpretation of one deterministic schema change."""

    change_type: ChangeType
    column: str
    compatibility: Compatibility = Field(description="System-verified against the deterministic diff.")
    impact: str = Field(max_length=1500, description="Consequences for downstream consumers.")
    recommendation: str = Field(max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=10, description="At least one evidence id.")
    status: ClaimStatus = "confirmed"
    grounding_notes: list[str] = Field(default_factory=list, description="System field; leave empty.")

    _cite = field_validator("evidence_ids")(_require_citation)


class SchemaAnalysisReport(_Model):
    """Schema Analyst output."""

    summary: str = Field(max_length=3000)
    drift_detected: bool = Field(description="System field; overwritten from the deterministic diff.")
    assessments: list[SchemaChangeAssessment] = Field(default_factory=list, max_length=50)
    confidence_estimate: float = Field(ge=0, le=1, description=_CONFIDENCE)
    limitations: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------- root cause


class Hypothesis(_Model):
    """A claim with explicit supporting and contradicting evidence."""

    statement: str = Field(max_length=1000)
    status: Literal["confirmed", "hypothesis", "rejected"]
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    contradicting_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    confidence_estimate: float = Field(ge=0, le=1, description=_CONFIDENCE)
    how_to_verify: str | None = Field(default=None, max_length=500)
    grounding_notes: list[str] = Field(default_factory=list, description="System field; leave empty.")


class RootCause(_Model):
    """A likely cause linking one or more findings."""

    cause_id: str = Field(max_length=20)
    category: Literal["upstream_source", "ingestion_parsing", "schema_change", "data_entry", "business_process",
                      "transformation_bug", "unknown"]
    explanation: str = Field(max_length=2000)
    related_finding_ids: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=10)


class ImpactAssessment(_Model):
    """Downstream impact."""

    affected_columns: list[str] = Field(default_factory=list)
    affected_rows_pct: float | None = Field(default=None, ge=0, le=100, description="Must equal a number in cited evidence.")
    downstream_risks: list[str] = Field(default_factory=list, max_length=10)
    blast_radius: Literal["low", "medium", "high"]
    evidence_ids: list[str] = Field(default_factory=list)
    grounding_notes: list[str] = Field(default_factory=list, description="System field; leave empty.")


class FollowUpRequest(_Model):
    """A targeted question for the Data Investigator."""

    question: str = Field(max_length=500)
    column: str | None = None
    suggested_tool: str | None = None


class RootCauseReport(_Model):
    """Root Cause Analyst output."""

    summary: str = Field(max_length=3000)
    confirmed_facts: list[Hypothesis] = Field(default_factory=list, max_length=20)
    root_causes: list[RootCause] = Field(default_factory=list, max_length=10)
    impact: ImpactAssessment
    needs_more_evidence: list[FollowUpRequest] = Field(default_factory=list, max_length=3)
    confidence_estimate: float = Field(ge=0, le=1, description=_CONFIDENCE)
    limitations: list[str] = Field(default_factory=list, max_length=10)
    unverified_claims: list[Hypothesis] = Field(default_factory=list, description="System field; leave empty.")


# ---------------------------------------------------------------------- remediation


class ProposedFix(_Model):
    """A proposed transformation. Never executed without human approval."""

    fix_id: str = Field(max_length=20, description="Unique id, e.g. 'FIX-1'.")
    addresses_finding_ids: list[str] = Field(min_length=1)
    title: str = Field(max_length=200)
    strategy: Literal["filter", "impute", "cast", "deduplicate", "standardize", "quarantine", "clip", "nullify"]
    sql: str = Field(max_length=10_000, description="One DuckDB SELECT over the relation `dataset`, returning ALL columns.")
    pyspark: str = Field(max_length=10_000, description="Equivalent PySpark: transform `df`, assign to `result`.")
    explanation: str = Field(max_length=2000)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    risk_level: Literal["low", "medium", "high"]
    reversible: bool
    expected_effect: str = Field(max_length=1000)


class RemediationPlan(_Model):
    """Remediation Agent output."""

    fixes: list[ProposedFix] = Field(default_factory=list, max_length=10)
    execution_order: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)


# ---------------------------------------------------------------------- system records


class FixValidation(_Model):
    """Deterministic validation of a proposed fix (system)."""

    fix_id: str
    sql_sha256: str
    sql_ok: bool
    sql_violations: list[str] = Field(default_factory=list)
    pyspark_ok: bool
    pyspark_violations: list[str] = Field(default_factory=list)
    dry_run_ok: bool = False
    dry_run_error: str | None = None
    columns_preserved: bool = False
    rows_before: int | None = None
    rows_after: int | None = None
    score_before: float | None = None
    score_after: float | None = None
    failed_checks_before: int | None = None
    failed_checks_after: int | None = None

    @property
    def approvable(self) -> bool:
        """Only fixes passing every deterministic gate may be offered for approval."""
        return self.sql_ok and self.pyspark_ok and self.dry_run_ok and self.columns_preserved


class ApprovalDecision(_Model):
    """A human decision bound to the exact SQL hash (system)."""

    fix_id: str
    sql_sha256: str
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str | None = Field(default=None, max_length=1000)
    decided_at: str
    system_note: str | None = None


class FixExecution(_Model):
    """Execution outcome of one fix (system)."""

    fix_id: str
    status: Literal["executed", "blocked", "failed"]
    reason: str
    sql_sha256: str
    rows_before: int | None = None
    rows_after: int | None = None


class ExecutionResult(_Model):
    """Outcome of the execution node (system)."""

    records: list[FixExecution] = Field(default_factory=list)
    artifact_name: str | None = None
    rows_before: int
    rows_after: int | None = None


class VerificationResult(_Model):
    """Before/after quality comparison computed by the engine (system)."""

    score_before: float | None
    score_after: float | None
    grade_before: str
    grade_after: str
    failed_checks_before: int
    failed_checks_after: int
    dimension_deltas: dict[str, float | None] = Field(default_factory=dict)
    resolved_checks: list[str] = Field(default_factory=list)
    new_failures: list[str] = Field(default_factory=list)


class AgentRunStatus(_Model):
    """Lifecycle of one agent run (system)."""

    status: Literal["pending", "succeeded", "degraded", "failed", "skipped"]
    tool_calls: int = 0
    repairs: int = 0
    message: str = ""


class AgentError(_Model):
    """A recorded, non-fatal problem (system)."""

    agent: str
    kind: Literal["llm", "tool", "validation", "grounding", "budget", "execution", "internal"]
    message: str
    timestamp: str


class TokenUsage(_Model):
    """LLM usage per agent (system)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class FailedCheckDigest(_Model):
    """A failed quality check as shown to agents."""

    evidence_id: str
    rule_id: str
    column: str | None
    severity: str
    affected_rows: int
    affected_pct: float
    message: str


class ColumnDigest(_Model):
    """Column schema summary as shown to agents."""

    name: str
    semantic_type: str
    storage_type: str
    null_count: int
    unique_count: int


class QualityDigest(_Model):
    """Compact, privacy-filtered summary of the deterministic analysis (system)."""

    dataset_name: str
    rows: int
    columns: int
    overall_score: float | None
    grade: str
    dimension_scores: dict[str, float | None]
    failed_checks: list[FailedCheckDigest]
    schema_columns: list[ColumnDigest]
    primary_key: str | None
    privacy_mode: str
    has_baseline: bool
    drift_summary: dict[str, int] | None = None


class InvestigationReport(_Model):
    """Final report. Quality metrics come from the engine, never from LLM text (system)."""

    run_id: str
    dataset_name: str
    privacy_mode: str
    generated_at: str
    quality_score: float | None
    grade: str
    findings: list[Finding] = Field(default_factory=list)
    schema_analysis: SchemaAnalysisReport | None = None
    root_cause: RootCauseReport | None = None
    remediation: RemediationPlan | None = None
    validations: list[FixValidation] = Field(default_factory=list)
    approvals: list[ApprovalDecision] = Field(default_factory=list)
    execution: ExecutionResult | None = None
    verification: VerificationResult | None = None
    agent_status: dict[str, AgentRunStatus] = Field(default_factory=dict)
    incomplete_agents: list[str] = Field(default_factory=list, description="Agents that failed or degraded.")
    errors: list[AgentError] = Field(default_factory=list)
    evidence_count: int = 0
    outcome: str
    disclaimer: str = (
        "Findings are produced by LLM agents reasoning over deterministic evidence. Confidence values are "
        "uncalibrated LLM estimates. Claims marked 'hypothesis' are not verified."
    )


CHECKPOINT_MODELS = (
    Evidence, Finding, InvestigatorReport, SchemaChangeAssessment, SchemaAnalysisReport, Hypothesis, RootCause,
    ImpactAssessment, FollowUpRequest, RootCauseReport, ProposedFix, RemediationPlan, FixValidation,
    ApprovalDecision, FixExecution, ExecutionResult, VerificationResult, AgentRunStatus, AgentError, TokenUsage,
    FailedCheckDigest, ColumnDigest, QualityDigest, InvestigationReport,
)
"""Models allowed to be deserialised from graph checkpoints (strict allow-list)."""
