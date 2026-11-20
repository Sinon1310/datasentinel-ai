# DataSentinel AI

**Agentic AI Data Engineering and Pipeline Observability Platform.**

## Overview

DataSentinel AI helps data teams understand and trust their datasets before those datasets
enter a pipeline. Its core rule is a strict separation of responsibilities: **deterministic code
computes the facts** (schema, statistics, rule violations, scores), and, from Milestone 2,
**LLM agents reason over those facts** to explain issues and recommend fixes. Agents never
compute the numbers themselves.

Milestone 1 delivers the **Data Intelligence Engine**: validated dataset ingestion, schema
inference, statistical profiling, a deterministic data quality engine with a transparent
score, and a Streamlit dashboard on top.

| Milestone | Scope | Status |
|---|---|---|
| 1 | Data Intelligence Engine: ingestion, schema, profiling, quality, dashboard | ✅ Complete |
| 2 | LangGraph agents (Groq / Llama) reasoning over engine output | Planned |

## Technology stack

| Layer | Technology | Used in Milestone 1 |
|---|---|---|
| Language | Python (3.11+ target; code is 3.9+ compatible) | ✅ |
| Data processing | pandas, NumPy | ✅ |
| Configuration | pydantic, pydantic-settings, python-dotenv | ✅ |
| Dashboard | Streamlit, Plotly | ✅ |
| Testing | pytest, `streamlit.testing` | ✅ |
| Agents and LLM | LangChain, LangGraph, Groq (Llama models) | Planned |
| API | FastAPI, Uvicorn | Planned |
| Scale-out processing | DuckDB, PySpark | Planned |
| Storage and retrieval | SQLite, ChromaDB | Planned |

See [docs/architecture.md](docs/architecture.md) for the detailed architecture and
[CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

---

## Quick start

```bash
# 1. Environment (Python 3.11+ recommended; the code is compatible with 3.9+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # full stack (Milestone 1 + planned milestones)
# or, Milestone 1 only:  pip install -e ".[dev]"
cp .env.example .env             # optional; no keys are needed for Milestone 1

# 2. Sample data (synthetic and seeded, included in the repo; regenerate any time)
python scripts/generate_sample_data.py

# 3. Run the dashboard
streamlit run app/streamlit_app.py      # http://localhost:8501

# 4. Run the tests
python -m pytest                     # full suite (engine + headless dashboard tests)
python -m pytest tests/test_quality.py -v   # a single module
```

The sample datasets are fully synthetic. They come from `scripts/generate_sample_data.py` with a
fixed random seed, and all email addresses use the reserved `example.com` domain.

Use the engine from Python without the UI:

```python
from datasentinel.pipeline import analyze_file

result = analyze_file("sample_data/customer_orders.csv")
print(result.quality.overall_score, result.quality.grade)
print(result.quality.results_dataframe().query("status == 'failed'"))
```

---

## Project layout

```
datasentinel/                 # Engine package — no UI, no LLM code
├── config.py                 # pydantic-settings; env vars prefixed DATASENTINEL_
├── logging_config.py         # namespaced logger setup
├── exceptions.py             # DataSentinelError hierarchy
├── pipeline.py               # ingest → infer → cast → profile → validate
├── ingestion/                # DatasetLoader (CSV / JSON / JSON Lines) + metadata
├── schema/                   # SchemaInferrer, SchemaReport, SchemaContract
├── profiling/                # DataProfiler + profile models
├── quality/                  # rules, config, scoring, engine, report models
└── utils/                    # shared parsers and JSON serialisation
app/                          # Streamlit presentation layer
├── streamlit_app.py          # entry point (sidebar, caching, error handling, tabs)
└── components/               # styles, plotly chart builders, page sections
tests/                        # pytest suite (engine + headless UI smoke tests)
scripts/generate_sample_data.py
sample_data/                  # customer_orders.csv, sensor_readings.json
```

---

## Features

### A. Ingestion (`datasentinel.ingestion`)
- CSV, JSON (record arrays, `{"data": [...]}` wrappers, columnar objects, single objects) and JSON Lines.
- Extension and size validation **before** reading (size limit from `DATASENTINEL_MAX_UPLOAD_SIZE_MB`).
- Rejects empty, header-only, binary and malformed files with specific exceptions
  (`UnsupportedFileTypeError`, `FileSizeError`, `EmptyDatasetError`, `DatasetParseError`).
- Recovers where safe and records a warning: delimiter sniffing (`, ; \t |`), UTF-8/BOM → cp1252 → latin-1
  fallback, malformed CSV rows skipped, phantom trailing-delimiter columns dropped, nested JSON flattened and
  list values serialised.

### B. Schema inference (`datasentinel.schema`)
- Semantic types: `numeric`, `categorical`, `datetime`, `boolean`, `text`, `empty`, plus a storage type.
- Typed values hidden in strings are detected by parse-success ratio (default ≥ 90%; ≥ 60% for date-named
  columns such as `created_at`), so a few dirty values don't flip a column's type. They are reported as
  invalid instead.
- Codes with leading zeros (ZIP codes, for example) stay text. Whole-number floats and 0/1 flags get a note.
- Primary-key candidates with confidence and evidence; composite (two-column) keys when no single column is unique.
- **Schema contracts**: export the inferred schema as an editable JSON contract, then upload it to validate
  future loads (types, required, nullable, unique, extra columns).

### C. Profiling (`datasentinel.profiling`)
- Rows, columns, total and missing cells, duplicate rows, memory footprint.
- Per column: null count and %, unique count and %, constant and unique flags.
- Numeric: min, max, mean, median, std, p05/q1/q3/p95, IQR, skewness, kurtosis, zeros, negatives, histogram.
- Categorical/text: cardinality, top-N frequencies, mode, Shannon entropy (raw and normalised), string lengths.
- Datetime: range and span. Boolean: true/false counts.

### D. Data quality engine (`datasentinel.quality`)

| Rule id | Dimension | What it checks | Severity bands (affected %) |
|---|---|---|---|
| `completeness.missing_values` | completeness | nulls + blank strings per column | ≤5 low · ≤20 medium · ≤50 high · >50 critical |
| `uniqueness.duplicate_records` | uniqueness | exact duplicate rows | ≤1 low · ≤5 medium · ≤20 high · >20 critical |
| `validity.data_types` | validity | values that don't parse as the inferred type; mixed JSON types | ≤1 low · ≤5 medium · ≤20 high · >20 critical |
| `validity.dates` | validity | unparseable or impossible dates (`2024-02-30`), before 1900, more than 5 years ahead | as above |
| `accuracy.outliers_iqr` | accuracy | Tukey fences (1.5×IQR mild, 3×IQR extreme) | ≤5 low · ≤15 medium · >15 high |
| `consistency.schema` | consistency | contract validation, or structural checks (empty/constant columns, bad headers, no key) | per sub-check |

Each result carries status (`passed` / `failed` / `skipped`), severity, affected rows and %, threshold,
evidence `details` and sample offending values. Every threshold lives in `QualityConfig`.

**Scoring (fully reproducible):**
1. Check score = `100 × (1 − penalty)`, with penalties low 0.25, medium 0.5, high 0.75, critical 1.0. A passed check scores 100 and skipped checks are excluded.
2. Dimension score = mean of its check scores.
3. Overall = weighted mean of dimension scores (completeness 25%, uniqueness 20%, validity 25%,
   consistency 15%, accuracy 15%). Weights are re-normalised over dimensions that had evaluated checks.
4. Grade: A ≥ 90 · B ≥ 80 · C ≥ 70 · D ≥ 60 · F.

The report stores each dimension's effective weight and contribution. The contributions sum to the overall score.

### E. Dashboard (`app/`)
- KPIs: quality score, grade, rows, columns, failed checks, missing cells.
- **Overview**: file metadata, ingestion warnings, missing-value chart, type distribution, severity breakdown, top issues.
- **Schema Explorer**: primary-key candidates, filterable column schema, schema/contract JSON downloads.
- **Data Quality**: score breakdown table and methodology, dimension chart, filterable rule results, evidence per failed check, JSON/CSV report downloads.
- **Profiling**: per-column explorer (histogram with IQR fences, box plot, top values, time trend), correlation heatmap.
- **Data Preview**: raw vs typed view, filters for rows with missing values or duplicates.
- Sidebar: upload, bundled samples, schema-contract upload, IQR multiplier and missing-value tolerance.

---

## Architectural decisions

1. **Deterministic engine, separate presentation and (future) reasoning.** `datasentinel/` never imports
   Streamlit or an LLM. The UI calls `pipeline.analyze_dataset`. In Milestone 2, LangGraph agents will
   call the same function as a tool and reason over its structured output. They interpret results; they
   never compute them.
2. **Raw vs typed frames.** Ingestion keeps the data exactly as delivered. `SchemaInferrer.cast` produces a
   typed copy where non-conforming values become null. Invalid-value detection is then just
   *present in raw but null in typed*, so inference and validation can't disagree because they share
   the same parsers (`utils/parsing.py`).
3. **Threshold-based inference.** A column that is 99% numbers is numeric with 1% invalid values. It is not
   treated as text. This is what makes "invalid data type" findings meaningful.
4. **Rules as pluggable classes.** Each rule is a small `QualityRule` subclass returning `RuleResult`s.
   The engine isolates failures: a rule that raises an exception becomes a `skipped` result and is logged,
   and the rest of the report is still produced.
5. **Explainable scoring over clever scoring.** A simple severity-penalty model with recorded weights and
   per-check scores, so anyone can recompute the score by hand. Thresholds are configuration, not code.
6. **Typed dataclasses + `to_builtin`.** Every result serialises to JSON, ready for the FastAPI layer,
   SQLite persistence and LLM prompts in later milestones.
7. **Secrets via environment only.** `Settings` reads `GROQ_API_KEY` from env / `.env` as a `SecretStr`.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATASENTINEL_MAX_UPLOAD_SIZE_MB` | `200` | Ingestion size limit (also set `server.maxUploadSize` in `.streamlit/config.toml`) |
| `DATASENTINEL_LOG_LEVEL` | `INFO` | Log level for the `datasentinel` logger |
| `GROQ_API_KEY` | — | Reserved for Milestone 2 |

## Known limitations (Milestone 1)
- Files are read fully into memory with pandas. Very large files will need the DuckDB/PySpark backends
  planned for later milestones.
- Excel, Parquet and UTF-16 files are not supported yet.
- Composite key search is limited to column pairs among the first 20 eligible columns.
- Out-of-range dates beyond pandas' nanosecond bounds (after 2262) are reported as unparseable rather than as too far in the future.
- No persistence yet. Reports can be downloaded but are not stored in SQLite.
