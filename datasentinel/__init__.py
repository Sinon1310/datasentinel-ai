"""DataSentinel AI — Agentic AI Data Engineering and Pipeline Observability Platform.

Milestone 1 provides the deterministic *Data Intelligence Engine*:

* :mod:`datasentinel.ingestion` — validated CSV / JSON loading
* :mod:`datasentinel.schema` — schema inference, primary-key detection, schema contracts
* :mod:`datasentinel.profiling` — statistical profiling
* :mod:`datasentinel.quality` — rule-based data quality validation and scoring
* :mod:`datasentinel.pipeline` — a single entry point that chains the above

None of these modules call an LLM. Later milestones layer LLM reasoning (LangGraph
agents) *on top of* the structured outputs produced here.
"""

__version__ = "0.1.0"
