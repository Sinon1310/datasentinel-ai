# DataSentinel AI — Milestone 2: Multi-Agent Investigation Architecture

Status: **approved and implemented**. This document is the design of record for the agent layer.
Milestone 1's deterministic engine is described in [architecture.md](architecture.md).

## 1. Approved decisions

| # | Decision | Consequence in the code |
|---|---|---|
| 1 | Add `sqlglot` and `langgraph-checkpoint-sqlite` | SQL safety uses a real parser; graph checkpoints persist in SQLite |
| 2 | PySpark is generated and **statically validated only** | `datasentinel/remediation/pyspark_check.py` uses AST analysis; nothing ever executes generated Python |
| 3 | **Metadata-only** is the default Groq privacy mode | Masked sample rows require an explicit per-run opt-in (`privacy_mode="masked_samples"`) |
| 4 | Baselines come from uploaded schema contracts **or** previously exported schema JSON | `datasentinel/schema/drift.py` normalises both into a `SchemaContract` |
| 5 | Separate top-level `datasentinel_agents/` package | Strict direction: `datasentinel_agents → datasentinel`. The engine never imports LLM code (enforced by a test) |
| 6 | Python 3.12 | Verified in a new environment (`.venv312`) before the old one is retired. All existing tests pass unchanged |

### Additional requirements and where they are met

| Req. | Requirement | Implementation |
|---|---|---|
| A | Checkpoints and datasets survive restarts | `SqliteSaver` checkpoints (`thread_id = run_id`), plus a `RunStore` that persists the source bytes, settings and SHA-256 per run. `Workspace` is rebuilt deterministically on resume |
| B | Evidence grounding | Every tool call becomes an `Evidence` record. `grounding.py` drops unknown ids, downgrades unsupported "confirmed" claims to hypotheses, and checks structured numbers against the cited evidence. Quality metrics in reports are copied from the engine, never taken from LLM text |
| C | Confidence is an estimate | Field named `confidence_estimate`, with `limitations` kept next to it. The UI labels it "LLM estimate (uncalibrated)" |
| D | Approval integrity | Approval is bound to `sha256(sql)`. The hash and the SQL policy are re-validated immediately before execution. Output is a new artefact; the source is never overwritten. Append-only JSONL audit log |
| E | Security | Privacy guard on every LLM input. DuckDB runs with `enable_external_access=false` and `lock_configuration=true`. All SQL is validated with sqlglot. PySpark is never executed. Dataset values are delimited, truncated, tokenised or masked, and the prompts declare them inert |

## 2. Layering

```
app/ (Streamlit)  ──▶  datasentinel_agents.service.InvestigationService
                              │
datasentinel_agents/          ▼          LLM layer: prompts, agent loop, LangGraph, Pydantic models
   graph · nodes · tools · grounding · evidence · llm · models · state
                              │  plain function calls (read-only tools)
datasentinel/                 ▼          deterministic engine: no LLM imports, ever
   EXISTING  ingestion · schema · profiling · quality · pipeline
   NEW       schema/drift · investigation/{workspace, drilldown, privacy} · remediation/{sql_guard, sandbox, pyspark_check, executor, audit}
```

## 3. Agents

| Agent | Input (from state) | Tools | Output |
|---|---|---|---|
| Data Investigator | quality digest, seeded evidence, follow-up requests | quality summary, failed checks, check details, column profile/schema, issue samples, issue rate by segment, issue trend, missingness co-occurrence | `InvestigatorReport` |
| Schema Analyst | deterministic `SchemaDiff`, seeded evidence | schema diff, inferred schema, baseline contract, column profile/schema, rename candidates | `SchemaAnalysisReport` |
| Root Cause Analyst | both grounded reports | evidence lookup, list findings, segment rate and trend (to test hypotheses) | `RootCauseReport` |
| Remediation | root-cause report (confirmed facts), findings, schema | check details, column schema/profile, `dry_run_sql` | `RemediationPlan` (SQL + PySpark) |

No agent has an execution tool. Execution is a deterministic graph node that runs only after approval.

## 4. Execution flow

```
START → prepare_context → route ─┬─ nothing to investigate ───────────────────────────────▶ report → END
                                 ├─▶ data_investigator ─┐ (parallel when both apply)
                                 └─▶ schema_analyst ────┴─▶ grounding_check → root_cause_analyst
            root_cause_analyst ─┬─ follow-up requested & rounds < 1 ─▶ data_investigator
                                ├─ no confirmed, remediable findings ─────────────────────▶ report
                                └─▶ remediation_agent → validate_fixes ─┬─ none valid & repairs < 1 ─▶ remediation_agent
                                                                        ├─ none valid ─────────────────▶ report
                                                                        └─▶ human_approval (interrupt)
            human_approval ─┬─ nothing approved ─────▶ report
                            └─▶ execute_fixes → verify → report → END
```

## 5. Evidence flow

1. `prepare_context` seeds the ledger from engine output: one `E-Q-NNN` per failed quality check and one `E-S-NNN` per schema-diff item.
2. Each agent's tool calls append `E-<AGENT>-NNN` records: tool name, arguments, privacy-filtered result. Prefixes keep ids unique across parallel branches.
3. Agents cite ids. Submissions are validated with a submission context that requires at least one citation on findings and schema assessments. After grounding removes invalid ids, a claim may be stored with zero citations, but only as a `hypothesis`. Stored records therefore always remain valid, which keeps checkpoint deserialisation exact.
4. `grounding_check` (deterministic) enforces requirement B. Notes on every downgrade are kept in the report.
5. Agents communicate only through typed state fields (blackboard). Their message histories are private.

## 6. Privacy modes

| Mode | Sent to Groq | Never sent |
|---|---|---|
| `metadata_only` (default) | schema, aggregate statistics, counts, rates, rule results, value *shape signatures* (`"99,99"`), tokenised category labels (`<region#3>`) | raw cell values, sample rows, free text |
| `masked_samples` (opt-in) | the above, plus up to N sample values with emails, phone numbers and long digit runs masked, truncated to 64 characters | unmasked PII patterns |

Tokens are deterministic per run and rehydrated to real labels only in the local UI.

## 7. Error handling

| Failure | Handling |
|---|---|
| Tool argument or lookup error | Returned to the model as an error message; counts toward the tool budget |
| Groq 429/5xx/timeout/connection | `RetryPolicy` on agent nodes (exponential backoff with jitter, 3 attempts, Groq transient errors only). `ChatGroq(max_retries=0)` so retries don't double up. If retries are exhausted, the run stops at its last checkpoint with status `incomplete`, and `InvestigationService.retry()` continues from there |
| Invalid structured output | Repair loop with the validation errors (max 2), then the agent is marked `degraded` |
| Agent failure | `agent_status=failed` plus an `AgentError`. The graph continues with partial results |
| Unsafe or invalid SQL | One remediation repair round. Fixes that are still invalid never reach approval |
| Execution error | Per-fix failure recorded. The source dataset is untouched |
| Process restart | Resume from the SQLite checkpoint, with the workspace rebuilt from the `RunStore` |

## 8. LLM configuration and structured output

- Model: `GROQ_MODEL` (default `openai/gpt-oss-120b`; legacy alias `DATASENTINEL_GROQ_MODEL`; optional per-deployment
  override `DATASENTINEL_AGENT_MODEL`). The model must support tool calling.
- Structured output uses **ordinary function tools**, not Groq's strict structured outputs. Groq currently cannot
  combine strict structured outputs (`response_format` json_schema) with tool use in the same request, so the agent loop
  never sends `response_format` or `strict`. Each agent finishes by calling the `submit_report` function tool, whose
  parameter schema is the Pydantic output model. When the tool budget is spent, the loop forces that tool with a standard
  `tool_choice`. Validation happens locally: Pydantic with the submission context, then the repair loop, then
  deterministic grounding. A test asserts the exact request shape.

## 9. Persistence layout

```
.datasentinel/                       (DATASENTINEL_DATA_DIR, git-ignored)
├── checkpoints.sqlite               LangGraph SqliteSaver (thread_id = run_id), strict deserialisation allow-list
└── runs/<run_id>/
    ├── manifest.json                filename, SHA-256, privacy mode, baseline, quality settings
    ├── source.<ext>                 original bytes (verified by hash on every rebuild)
    ├── artifacts/remediated.parquet new files only, never overwritten
    └── audit.jsonl                  append-only approval and execution events
```

## 10. Known limitations

- Generated PySpark is never executed, so SQL/PySpark equivalence is not verified.
- Dry runs use the first `dry_run_rows` rows. Verification after execution uses the full dataset.
- No fallback model is configured. A Groq outage leaves a resumable `incomplete` run.
- Category-label tokens are deterministic per dataset, but the LLM never learns real values in metadata-only mode, so semantic reasoning about labels is limited by design.
- Runs execute in the calling process (no job queue). The Streamlit UI blocks while agents run.

## 11. Testing

- Deterministic modules are tested like Milestone 1.
- Agents and the graph are tested with a scripted fake chat model: tool calling, repair, budget, routing, parallel fan-out, retries, interrupt/resume across a new `SqliteSaver` connection, and the approval hash checks.
- Security tests cover the SQL attack corpus, PySpark AST violations, prompt-injection values and approval tampering.
- An import-direction test enforces that `datasentinel/` never imports agent or LLM packages.
- Live Groq tests are opt-in: `pytest -m live` with `GROQ_API_KEY` set.
