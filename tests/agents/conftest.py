"""Fixtures for agent-layer tests (no network; scripted LLMs)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from datasentinel.config import Settings
from datasentinel_agents.config import AgentSettings
from datasentinel_agents.service import InvestigationService
from datasentinel_agents.services import Services
from tests.agents.fakes import FakeLLMFactory

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data" / "customer_orders.csv"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", _env_file=None)


@pytest.fixture
def agent_settings() -> AgentSettings:
    return AgentSettings(retry_initial_interval=0.0, _env_file=None)


@pytest.fixture
def make_service(settings: Settings, agent_settings: AgentSettings) -> Callable[..., tuple[InvestigationService, FakeLLMFactory]]:
    services_created: list[InvestigationService] = []

    def _make(scripts: dict | None = None, **overrides) -> tuple[InvestigationService, FakeLLMFactory]:
        fake = FakeLLMFactory(scripts or {})
        agent = agent_settings.model_copy(update=overrides) if overrides else agent_settings
        service = InvestigationService(Services.create(llm_factory=fake, settings=settings, agent_settings=agent))
        services_created.append(service)
        return service, fake

    yield _make
    for service in services_created:
        service.close()


@pytest.fixture
def sample_bytes() -> bytes:
    return SAMPLE.read_bytes()
