# DataSentinel AI — Milestone 1 Architecture

This document describes the architecture of the **Data Intelligence Engine** as implemented in
Milestone 1. It covers what exists today. Planned components are labelled as such.

Milestone 2 added deterministic modules to this engine (`schema/drift.py`, `investigation/`, `remediation/`)
and a separate LLM layer, `datasentinel_agents/`. Both are documented in [agents.md](agents.md). The engine still
imports no LLM or UI code, and `tests/test_architecture.py` enforces this. Target runtime: Python 3.12.

## 1. Goals and principles

| Principle | How it is enforced |
|---|---|
| Deterministic facts, separate from LLM reasoning | `datasentinel/` contains no LLM code. Future agents will call the pipeline as a tool. |
| Presentation is a thin layer | `app/` renders an `AnalysisResult`. It computes nothing itself. |
| Explainable results | Every check records its threshold, evidence and score contribution. |
| Graceful failure | Typed exceptions for expected errors. Recoverable issues become warnings. A failing rule is isolated. |
| Configuration over code | Thresholds live in `QualityConfig` / `InferenceConfig`. Runtime settings come from environment variables. |

## 2. System overview

```
                 ┌──────────────────────────────────────────────┐
  CSV / JSON ──▶ │  app/streamlit_app.py  (presentation layer)  │
  (upload or     │  sidebar nav · caching · errors · sections   │
   sample)       └───────────────────────┬──────────────────────┘
                                         │ bytes, filename, settings
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │  datasentinel.pipeline.analyze_dataset       │
                 └───────────────────────┬──────────────────────┘
                                         │
   ┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐
   │ ingestion   │──▶│ schema.infer │──▶│ schema.  │──▶│ profiling    │──▶│ quality      │
   │ DatasetLoader│  │ SchemaReport │   │ cast     │   │ DatasetProfile│  │ QualityReport│
   └─────────────┘   └──────────────┘   └──────────┘   └──────────────┘   └──────────────┘
     raw DataFrame                       typed DataFrame
                                         │
                                         ▼
                                AnalysisResult (dataclasses, JSON-serialisable)
```

## 3. Package layout

```
datasentinel/
├── config.py            Settings (pydantic-settings), env prefix DATASENTINEL_
├── logging_config.py    configure_logging(): handler on the "datasentinel" logger only
├── exceptions.py        DataSentinelError hierarchy
├── pipeline.py          analyze_dataset(), analyze_file(), AnalysisResult
├── ingestion/           loader.py (DatasetLoader), models.py (LoadedDataset, DatasetMetadata)
├── schema/              inference.py, models.py, contract.py
├── profiling/           profiler.py, models.py
├── quality/             rules.py, config.py, scoring.py, engine.py, models.py
└── utils/               parsing.py (shared coercion), serialization.py (to_builtin)
app/
├── streamlit_app.py     entry point
├── components/          shell (navigation), styles (tokens/CSS), ui (KPI cards, badges, states), dataset (session
│                        context + cached analysis), charts (pure Plotly builders), workflow, investigation
├── views/               overview, quality, schema, profiling, observability (one renderer per section)
├── observability/       typed UI contracts, not-connected live source, in-memory synthetic demo backend
└── pages/               1_Pipeline_Observability.py
```

**Dependency direction:** `app → datasentinel.pipeline → {ingestion, schema, profiling, quality} → utils`.
Engine packages never import `app` or `streamlit`. `quality` depends on `schema` and `profiling`
models. Nothing depends on `quality`.

## 4. Components

### 4.1 Ingestion — `DatasetLoader`

Entry points: `load()` (dispatches on source type), `load_path()`, `load_bytes()` (the core).

Processing order, cheapest checks first:

1. **Extension** must be in `Settings.allowed_extensions` (`.csv`, `.json`, `.jsonl`, `.ndjson`).
2. **Size**: zero bytes is rejected. Anything above `max_upload_size_mb` is rejected. For paths, the size is checked with `stat()` before reading.
3. **Binary sniff**: a NUL byte in the first 8 KB means the file is rejected.
4. **Decode**: UTF-8 (BOM-aware), then cp1252, then latin-1, with a warning on fallback.
5. **Parse**
   - CSV: the delimiter is sniffed from `, ; \t |`. If strict parsing fails, the loader retries in tolerant mode, which skips malformed rows and records a warning with the count and an example.
   - JSON: supports record arrays, wrapper objects (`data`, `records`, `rows`, `items`, `results`), columnar objects, single objects and JSON Lines. Nested objects are flattened with `json_normalize`.
6. **Finalise**: zero columns or zero rows is rejected. Column names are stripped and made unique, empty `Unnamed:` columns are dropped, and list/dict cells are serialised to JSON text so they stay hashable.

Output: `LoadedDataset(dataframe, metadata)`. The metadata records file type, size, encoding, delimiter,
JSON layout, row and column counts, load time and warnings.

Errors: `UnsupportedFileTypeError`, `FileSizeError`, `EmptyDatasetError`, `DatasetParseError`, all
subclasses of `IngestionError`.

### 4.2 Schema inference — `SchemaInferrer`

`infer(df) -> SchemaReport`, then `cast(df, report) -> typed DataFrame`.

Per-column decision procedure:

```
missing = null or whitespace-only string
if no present values           → EMPTY
elif bool dtype                → BOOLEAN
elif datetime dtype            → DATETIME
elif numeric dtype             → NUMERIC (integer | float; notes for whole-number floats, 0/1 flags)
elif categorical dtype         → CATEGORICAL
else (text), on a sample of up to 10,000 values:
    ≥95% boolean tokens (not only 0/1)   → BOOLEAN
    ≥90% parse as numbers, no leading 0s → NUMERIC
    ≥90% parse as dates (≥60% if the
      column name suggests a date)       → DATETIME
    average length ≥ 50                  → TEXT
    unique ratio ≤ 0.5, or ≤ 50 distinct
      values with unique ratio < 0.9     → CATEGORICAL
    otherwise                            → TEXT
```

**Primary keys:** a column qualifies when it has no nulls, every value is unique, and it is not
boolean, empty or a non-integral float. Confidence starts at 0.5 and gains or loses points for
identifier-style names, integer or categorical storage, first position (bonus) and timestamps (penalty).
If no single column qualifies, pairs among the first 20 eligible columns are tested as composite keys.

**Schema contracts** (`contract.py`): `SchemaContract` declares expected type, required, nullable and
unique per column, plus `allow_extra_columns`. It can be built from JSON (with validation that raises
`SchemaContractError`) or bootstrapped from an inferred `SchemaReport`.

### 4.3 Raw vs typed frames

`cast()` converts text-encoded numbers, dates and booleans with the shared parsers in
`utils/parsing.py`, and replaces blank strings with null. A value that cannot be converted becomes null.

```
invalid value  ⇔  present in raw  ∧  null in typed
```

Inference and validation use the same parsers, so they cannot disagree about a value's type.
The raw frame is kept for row-level facts (duplicates, exact missing counts). The typed frame
is used for statistics.

### 4.4 Profiling — `DataProfiler`

`profile(raw, typed, schema) -> DatasetProfile`

- Dataset level: rows, columns, total and missing cells, duplicate rows (from the raw frame), memory usage.
- Column level: null, unique and constant flags, plus statistics by type:
  - numeric: moments, quantiles, IQR, zeros, negatives, infinities and a histogram
  - categorical/text: cardinality, top-N, mode, entropy and string lengths
  - datetime: range and span
  - boolean: counts

### 4.5 Quality engine

```
QualityContext(raw, typed, schema, profile, contract?)  ─▶  QualityEngine.run()
      for rule in rules: results += rule.evaluate(ctx)    (exceptions → SKIPPED result)
      compute_scores(results)                              (fills result.score)
  ─▶  QualityReport(overall_score, grade, dimension_scores, results, methodology)
```

| Rule | Dimension | Granularity |
|---|---|---|
| `MissingValuesRule` | completeness | per column |
| `DuplicateRecordsRule` | uniqueness | dataset |
| `InvalidDataTypeRule` | validity | per column (numeric, boolean, text-like) |
| `InvalidDateRule` | validity | per datetime or date-named column |
| `OutlierIQRRule` | accuracy | per numeric column (skips PK, binary and IQR=0 columns) |
| `SchemaValidationRule` | consistency | contract checks, or structural checks |

Severity comes from the affected percentage through ordered bands in `QualityConfig`
(`classify_severity`).

**Scoring** (`scoring.py`):

```
check_score     = 100 × (1 − penalty[severity])      passed = 100, skipped = excluded
dimension_score = mean(check_scores in dimension)
overall         = Σ dimension_score × weight / Σ weight     (over evaluated dimensions only)
grade           = A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else F
```

Defaults:
- Penalties: low 0.25, medium 0.5, high 0.75, critical 1.0.
- Weights: completeness 0.25, uniqueness 0.20, validity 0.25, consistency 0.15, accuracy 0.15.

Each `DimensionScore` stores `effective_weight` and `contribution`. The contributions sum to the
overall score.

### 4.6 Presentation — Streamlit

- **Navigation:** one sidebar of `st.page_link` items (`components/shell.py`). The five dataset sections live in
  `streamlit_app.py` and are selected with `?section=` (overview, quality, schema, profiling, investigation);
  Pipeline Observability is `pages/1_Pipeline_Observability.py`. Streamlit's automatic page list is hidden
  (`.streamlit/config.toml`), so there is a single navigation system.
- **Theme:** native Streamlit theming in `.streamlit/config.toml` (light workspace, dark `[theme.sidebar]`);
  `components/styles.py` holds the same tokens for charts plus minimal CSS keyed on stable `data-testid`s.
- `streamlit_app.py` renders the shell and routes to `views/*`. The active dataset is kept in session state
  (`components/dataset.py`) so it survives page changes; analysis is the cached `run_analysis()`
  (`st.cache_data`, keyed on file bytes, baseline text and settings). Errors become explanations, never tracebacks.
- `components/charts.py`: pure functions that turn prepared data into Plotly figures; reserved status colours are
  always paired with text labels.
- `views/*`: one renderer per section, each with small pure view-model helpers (unit-tested).
- `components/investigation.py` + `components/workflow.py`: the AI Investigation workspace. Stage statuses are derived
  only from checkpointed workflow state; approvals go through `InvestigationService.resume`.
- `observability/`: the UI integration boundary for Milestone 3 — `ObservabilityDataSource` (read) and
  `ObservabilityCommands` (record-only retry request, cancel PENDING). Until the backend exists, the live source is
  `UnavailableBackend`; `DemoBackend` provides deterministic, clearly labelled synthetic data in memory only.

## 5. Cross-cutting concerns

| Concern | Implementation |
|---|---|
| Configuration | `Settings` via `get_settings()` (cached). Env vars use the `DATASENTINEL_` prefix, and `GROQ_API_KEY` is a `SecretStr`. |
| Logging | Module loggers with `getLogger(__name__)`. One handler on the `datasentinel` namespace, without touching the root logger. |
| Errors | `DataSentinelError` → `IngestionError` (4 subclasses), `SchemaContractError`. |
| Serialisation | Every model offers `to_dict()` via `to_builtin` (handles numpy, pandas, enum, NaN→null). |
| Testing | pytest: unit tests per package, hand-computed scoring assertions, headless UI tests via `streamlit.testing`. |

## 6. Extension points

- **New quality rule:** subclass `QualityRule`, register it in `default_rules()`, add thresholds to `QualityConfig`.
- **New file format:** add a parser branch in `DatasetLoader.load_bytes` and the extension to settings.
- **Custom thresholds:** pass `QualityConfig(...)` / `InferenceConfig(...)` to `analyze_dataset`.

## 7. Known limitations

- Files are processed fully in memory with pandas.
- Excel, Parquet and UTF-16 files are not supported.
- Composite keys are limited to pairs.
- Dates after 2262 (pandas' nanosecond bound) are reported as unparseable.
- Deterministic analysis results are not persisted. Investigation runs are, see [agents.md](agents.md).

## 8. Evolution

- **Milestone 2 (implemented):** multi-agent investigation. See [agents.md](agents.md).
- **Later (not implemented):**
  - a FastAPI service exposing the pipeline and `InvestigationService`
  - schema history in SQLite
  - ChromaDB for retrieval
  - PySpark execution and backends for datasets larger than memory
