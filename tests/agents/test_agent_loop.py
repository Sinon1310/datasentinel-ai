"""Tests for the shared tool-calling agent loop."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from datasentinel_agents.config import AgentSettings
from datasentinel_agents.llm import TransientLLMError
from datasentinel_agents.models import InvestigatorReport
from datasentinel_agents.runtime import SUBMIT_TOOL, AgentRunner
from tests.agents.fakes import ScriptedChatModel, submit, tool_call

VALID = {"summary": "ok", "findings": [{"finding_id": "F1", "title": "t", "issue_type": "missing", "severity": "low",
                                        "description": "d", "evidence_ids": ["E-Q-001"], "confidence_estimate": 0.4}]}


class EchoArgs(BaseModel):
    text: str


def _echo_tool(calls: list) -> StructuredTool:
    def echo(text: str) -> str:
        calls.append(text)
        return json.dumps({"evidence_id": f"E-INV-{len(calls):03d}", "data": {"echo": text}})

    return StructuredTool.from_function(func=echo, name="echo", description="echo", args_schema=EchoArgs)


def _run(script, settings: AgentSettings | None = None, tools=None):
    model = ScriptedChatModel(responses=list(script))
    runner = AgentRunner(model, settings or AgentSettings(_env_file=None))
    outcome = runner.run("data_investigator", "system", {"task": 1}, tools or [], InvestigatorReport, [])
    return outcome, model


def test_tool_then_submit() -> None:
    calls: list = []
    outcome, model = _run([tool_call("echo", {"text": "hi"}), submit(VALID)], tools=[_echo_tool(calls)])
    assert outcome.status.status == "succeeded" and outcome.output.findings[0].finding_id == "F1"
    assert calls == ["hi"] and outcome.status.tool_calls == 1
    assert SUBMIT_TOOL in model.bound_tools[0] and "echo" in model.bound_tools[0]
    tool_messages = [m for m in model.received[-1] if isinstance(m, ToolMessage)]
    assert json.loads(tool_messages[0].content)["evidence_id"] == "E-INV-001"


def test_invalid_submission_is_repaired() -> None:
    bad = {"summary": "x", "findings": [{**VALID["findings"][0], "evidence_ids": [], "confidence_estimate": 3}]}
    outcome, model = _run([submit(bad), submit(VALID)])
    assert outcome.status.status == "succeeded" and outcome.status.repairs == 1
    feedback = json.loads([m for m in model.received[1] if isinstance(m, ToolMessage)][-1].content)
    assert feedback["error"] == "validation_failed"
    assert any("evidence_ids" in item["loc"] for item in feedback["fix_these"])


def test_plain_text_answer_gets_reminder() -> None:
    outcome, model = _run([AIMessage(content="Here is my analysis..."), submit(VALID)])
    assert outcome.status.status == "succeeded" and outcome.status.repairs == 1
    assert "submit_report" in str(model.received[1][-1].content)


def test_degrades_after_repair_budget() -> None:
    outcome, _ = _run([AIMessage(content="no")] * 3, AgentSettings(max_repair_attempts=2, _env_file=None))
    assert outcome.output is None and outcome.status.status == "degraded"
    assert outcome.errors[-1].kind == "validation"


def test_tool_budget_forces_submission() -> None:
    calls: list = []
    script = [tool_call("echo", {"text": str(i)}) for i in range(2)] + [submit(VALID)]
    outcome, model = _run(script, AgentSettings(max_tool_calls=2, _env_file=None), tools=[_echo_tool(calls)])
    assert outcome.status.status == "succeeded" and len(calls) == 2
    # The first two calls had all tools; once the budget was spent only submit_report was offered, forced.
    assert [i["tool_choice"] for i in model.invocations] == [None, None, SUBMIT_TOOL]
    assert model.invocations[-1]["tools"] == [SUBMIT_TOOL]


def test_calls_beyond_budget_are_refused() -> None:
    calls: list = []
    many = AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": str(i)}, "id": f"c{i}"} for i in range(3)])
    outcome, model = _run([many, submit(VALID)], AgentSettings(max_tool_calls=2, _env_file=None), tools=[_echo_tool(calls)])
    assert len(calls) == 2 and outcome.status.status == "succeeded"
    refused = [m for m in model.received[1] if isinstance(m, ToolMessage)][-1]
    assert "budget" in refused.content


def test_unknown_tool_and_bad_arguments_return_errors() -> None:
    calls: list = []
    outcome, model = _run([tool_call("rm_rf", {}), tool_call("echo", {"wrong": 1}), submit(VALID)], tools=[_echo_tool(calls)])
    errors = [m.content for m in model.received[-1] if isinstance(m, ToolMessage)]
    assert "Unknown tool" in errors[0] and "error" in errors[1].lower()
    assert outcome.status.status == "succeeded" and calls == []


def test_transient_errors_propagate_for_retry_policy() -> None:
    with pytest.raises(TransientLLMError):
        _run([TransientLLMError("429")])
