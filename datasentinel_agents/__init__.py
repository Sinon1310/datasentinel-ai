"""DataSentinel AI — multi-agent investigation layer (Milestone 2).

LLM agents orchestrated by LangGraph that *interpret* the deterministic outputs of
:mod:`datasentinel`. Dependency direction is strictly ``datasentinel_agents → datasentinel``.

Entry point: :class:`datasentinel_agents.service.InvestigationService`.
"""

__version__ = "0.2.0"
