"""Tests for Groq model configuration and the request shape our agents send.

All tests isolate themselves from the developer's shell environment and local .env file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from datasentinel.config import DEFAULT_GROQ_MODEL, Settings
from datasentinel_agents.config import AgentSettings
from datasentinel_agents.llm import groq_llm_factory
from datasentinel_agents.models import InvestigatorReport
from datasentinel_agents.runtime.agent_loop import SUBMIT_TOOL, _submit_tool

MODEL_VARS = ("GROQ_MODEL", "DATASENTINEL_GROQ_MODEL", "DATASENTINEL_AGENT_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MODEL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy_key_for_config_tests")


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=kwargs.pop("_env_file", None), **kwargs)


class TestModelSelection:
    def test_default_model_is_gpt_oss_120b(self) -> None:
        assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"
        assert _settings().groq_model == "openai/gpt-oss-120b"

    def test_groq_model_env_var_overrides_default(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        assert _settings().groq_model == "openai/gpt-oss-20b"

    def test_legacy_variable_still_supported(self, monkeypatch) -> None:
        monkeypatch.setenv("DATASENTINEL_GROQ_MODEL", "qwen/qwen3.8-27b")
        assert _settings().groq_model == "qwen/qwen3.8-27b"

    def test_groq_model_wins_over_legacy_variable(self, monkeypatch) -> None:
        monkeypatch.setenv("DATASENTINEL_GROQ_MODEL", "llama-3.3-70b-versatile")
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
        assert _settings().groq_model == "openai/gpt-oss-120b"

    def test_reads_groq_model_from_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("GROQ_MODEL=openai/gpt-oss-20b\n")
        assert _settings(_env_file=env_file).groq_model == "openai/gpt-oss-20b"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_value_falls_back_to_default(self, monkeypatch, blank: str) -> None:
        monkeypatch.setenv("GROQ_MODEL", blank)
        assert _settings().groq_model == DEFAULT_GROQ_MODEL


class TestFactory:
    def test_factory_builds_chatgroq_with_configured_model(self) -> None:
        model = groq_llm_factory(_settings(), AgentSettings(_env_file=None))("data_investigator")
        assert isinstance(model, ChatGroq)
        assert model.model_name == "openai/gpt-oss-120b"
        assert model.max_retries == 0  # LangGraph RetryPolicy stays the single retry layer

    def test_agent_override_takes_precedence(self) -> None:
        agent = AgentSettings(model="openai/gpt-oss-20b", _env_file=None)
        assert groq_llm_factory(_settings(), agent)("schema_analyst").model_name == "openai/gpt-oss-20b"


def _captured_requests() -> list[dict]:
    """Invoke both bindings used by the agent loop and capture the exact Groq client arguments."""
    captured: list[dict] = []

    class RecordingClient:
        def create(self, **kwargs):
            captured.append(kwargs)
            raise RuntimeError("captured; no network")

    llm = groq_llm_factory(_settings(), AgentSettings(_env_file=None))("data_investigator")
    llm.client = RecordingClient()
    submit = _submit_tool(InvestigatorReport)
    for bound in (llm.bind_tools([submit]), llm.bind_tools([submit], tool_choice=SUBMIT_TOOL)):
        with pytest.raises(RuntimeError):
            bound.invoke([HumanMessage(content="x")])
    return captured


def test_submit_report_uses_plain_tool_calling_not_strict_structured_output() -> None:
    # Groq cannot combine strict structured outputs with tool use, so we must never send response_format/strict.
    normal, forced = _captured_requests()
    for request in (normal, forced):
        assert request["model"] == "openai/gpt-oss-120b"
        assert "response_format" not in request or request["response_format"] is None
        tool = request["tools"][0]
        assert tool["type"] == "function" and tool["function"]["name"] == SUBMIT_TOOL
        assert not tool["function"].get("strict")
        assert "$ref" not in str(tool["function"]["parameters"])  # fully inlined schema
        assert "evidence_ids" in str(tool["function"]["parameters"])  # Pydantic model drives the schema
    assert "tool_choice" not in normal or normal["tool_choice"] in (None, "auto")
    assert forced["tool_choice"] == {"type": "function", "function": {"name": SUBMIT_TOOL}}
