"""Dependency container passed to graph nodes (injected, so tests can swap the LLM)."""

from __future__ import annotations

from dataclasses import dataclass

from datasentinel.config import Settings, get_settings
from datasentinel.investigation.workspace import RunStore, WorkspaceManager
from datasentinel.remediation import AuditLog, DuckDBSandbox
from datasentinel_agents.config import AgentSettings, get_agent_settings
from datasentinel_agents.llm import LLMFactory, groq_llm_factory


@dataclass
class Services:
    """Runtime dependencies for an investigation graph."""

    settings: Settings
    agent_settings: AgentSettings
    store: RunStore
    workspaces: WorkspaceManager
    llm_factory: LLMFactory
    sandbox: DuckDBSandbox

    @classmethod
    def create(
        cls,
        llm_factory: LLMFactory | None = None,
        settings: Settings | None = None,
        agent_settings: AgentSettings | None = None,
    ) -> "Services":
        """Build services; uses Groq unless an ``llm_factory`` is supplied."""
        settings = settings or get_settings()
        agent_settings = agent_settings or get_agent_settings()
        store = RunStore(settings.data_dir)
        return cls(
            settings=settings,
            agent_settings=agent_settings,
            store=store,
            workspaces=WorkspaceManager(store),
            llm_factory=llm_factory or groq_llm_factory(settings, agent_settings),
            sandbox=DuckDBSandbox(),
        )

    def audit(self, run_id: str) -> AuditLog:
        """The append-only audit log of a run."""
        return AuditLog(self.store.audit_path(run_id))
