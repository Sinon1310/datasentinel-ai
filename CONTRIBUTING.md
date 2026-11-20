# Contributing to DataSentinel AI

Thank you for contributing. This guide explains how the project is organised, the conventions
every change is expected to follow, and how changes are tested.

## Development setup

```bash
git clone <repository-url> && cd datasentinel-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # never commit .env
python -m pytest              # confirm a green baseline before you start
```

Target Python 3.11+. Until the minimum version is formally raised, code must stay compatible with
**Python 3.9**:

- Start every module with `from __future__ import annotations`.
- Use `Optional[X]` / `Union[X, Y]` in any annotation evaluated at runtime (pydantic models, dataclass
  field defaults). Don't use `X | Y` there.
- No `match` statements, `StrEnum`, `kw_only` or `slots` dataclass options.

## Architecture rules

Read [docs/architecture.md](docs/architecture.md) first. The rules that matter most:

1. **Deterministic computation stays separate from LLM reasoning.** Nothing in `datasentinel/`
   may call an LLM. Agents (Milestone 2+) consume engine output; they never compute or override
   scores, statistics or rule results.
2. **The engine never imports the UI.** `datasentinel/` must not import `streamlit` or anything in `app/`.
   The UI calls `datasentinel.pipeline.analyze_dataset` and only renders the result.
3. **One responsibility per module.** Ingestion, schema, profiling and quality each live in their own
   package with a `models.py` for their data structures. Avoid catch-all modules.
4. **Shared parsing lives in `datasentinel/utils/parsing.py`.** Schema inference and quality
   validation must use the same parsers so they never disagree about a value's type.
5. **Thresholds are configuration.** Any number that affects pass/fail, severity or the score belongs in
   `QualityConfig` or `InferenceConfig`, not inline in rule logic.

## Coding conventions

- **Type hints** on all function signatures and dataclass fields.
- **Docstrings** (Google style) on every public module, class and function. Explain *why* when it
  isn't obvious from the code.
- **Logging:** use `logger = logging.getLogger(__name__)`. Never use `print` in library code. Log at
  `INFO` for stage summaries, `WARNING` for recovered problems, and use `logger.exception` for unexpected errors.
  Never log secrets or full row contents.
- **Exceptions:** raise a subclass of `DataSentinelError` (see `datasentinel/exceptions.py`) for every
  expected failure, with a message a user can act on. Chain the cause with `raise ... from exc`. Catch broad
  `Exception` only at isolation boundaries (the quality engine's rule loop, the UI's top-level guard).
- **Results are data.** Return typed dataclasses that serialise through `to_builtin`, so every result
  stays usable by the API, persistence and agents.
- **No hardcoded secrets.** Read configuration through `datasentinel.config.get_settings()`. Add new
  variables to `.env.example` with an empty value.
- **No mock outputs.** If a feature can be really implemented, implement it. Don't return placeholder data.
- Keep functions small and names descriptive. Prefer clarity over cleverness.

## Adding a quality rule

1. Subclass `QualityRule` in `datasentinel/quality/rules.py`. Set `rule_id` (`<dimension>.<name>`),
   `name`, `dimension` and `description`.
2. Implement `evaluate(ctx) -> List[RuleResult]`. Build results with `self.result(...)` and include
   evidence in `details` and `sample_values`.
3. Put thresholds and severity bands in `QualityConfig`.
4. Register the rule in `default_rules()`.
5. Add tests covering pass, fail (with severity) and skip paths.

## Testing expectations

- Run the full suite before opening a pull request: `python -m pytest`. **All tests must pass.**
- Every bug fix includes a regression test that fails without the fix.
- Every new feature includes unit tests for its core behaviour, edge cases (empty, null-only and
  single-row inputs) and error paths.
- Tests use real computation on small, explicit DataFrames or fixture files. Where possible, assert
  values you calculated by hand (see `TestScoring.test_hand_computed_score`).
- Put shared fixtures in `tests/conftest.py`. Tests write files only under pytest's `tmp_path`.
- UI changes must keep `tests/test_app.py` (headless `streamlit.testing` runs) green. Add a
  case if you add a new interactive path.

## Dependencies

`requirements.txt` and `pyproject.toml` must declare the same packages with the same minimum versions:

| `pyproject.toml` group | `requirements.txt` block |
|---|---|
| `[project].dependencies` | Runtime |
| `optional-dependencies.dev` | Testing |
| `optional-dependencies.platform` | Later milestones |

Explain any new dependency in your pull request.

## Commits and pull requests

- Branch from `main` (`feature/<topic>`, `fix/<topic>`, `docs/<topic>`).
- Write imperative commit subjects (`Add JSON Lines ingestion`), each covering one logical change.
- In the PR description, explain what changed and why, how you tested it, and any architectural
  decisions or trade-offs.
- Update `README.md` and `docs/architecture.md` when behaviour or structure changes.

## Data and privacy

Only commit **synthetic** data. Regenerate sample data with `scripts/generate_sample_data.py`
(fixed seed, `example.com` addresses). Never commit real customer data, credentials or `.env` files.
