"""A scripted, tool-calling fake chat model for deterministic agent and graph tests."""

from __future__ import annotations

import itertools
import json
import threading
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count(1)
Response = AIMessage | BaseException | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns scripted responses in order. Items may be messages, exceptions to raise, or callables."""

    responses: list[Any] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)
    tool_choices: list[Any] = Field(default_factory=list)
    invocations: list[dict[str, Any]] = Field(default_factory=list)
    active_tools: list[str] | None = None
    active_choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "ScriptedChatModel":
        names = [getattr(t, "name", str(t)) for t in tools]
        self.bound_tools.append(names)
        self.tool_choices.append(tool_choice)
        # Shallow copy shares the script and the recording lists, but remembers its own binding.
        return self.model_copy(update={"active_tools": names, "active_choice": tool_choice})

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        self.invocations.append({"tools": self.active_tools, "tool_choice": self.active_choice})
        if not self.responses:
            raise AssertionError("ScriptedChatModel: script exhausted")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        message = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=message)])


class FakeLLMFactory:
    """Per-agent scripted models; thread-safe for parallel graph branches."""

    def __init__(self, scripts: dict[str, list[Response]]) -> None:
        self.models = {agent: ScriptedChatModel(responses=list(items)) for agent, items in scripts.items()}
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(self, agent: str) -> ScriptedChatModel:
        with self._lock:
            self.calls.append(agent)
            if agent not in self.models:
                raise AssertionError(f"No script for agent '{agent}'")
            return self.models[agent]

    def all_text_sent(self) -> str:
        """Every message content sent to any model (for privacy assertions)."""
        return "\n".join(str(m.content) for model in self.models.values() for msgs in model.received for m in msgs)


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """An AI message calling one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def submit(report: dict[str, Any]) -> AIMessage:
    """An AI message submitting a final report."""
    return tool_call("submit_report", report)


def task_of(messages: list[BaseMessage]) -> dict[str, Any]:
    """The JSON task sent to the agent."""
    human = next(m for m in messages if isinstance(m, HumanMessage) and str(m.content).startswith("Task (JSON):"))
    return json.loads(str(human.content).split("\n", 1)[1])


def tool_evidence_ids(messages: list[BaseMessage]) -> list[str]:
    """Evidence ids returned by tools so far, in order."""
    ids = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                payload = json.loads(str(message.content))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "evidence_id" in payload:
                ids.append(payload["evidence_id"])
    return ids


def seeded_id(messages: list[BaseMessage], rule_id: str, column: str | None) -> str:
    """Seeded evidence id for a failed check, read from the task digest."""
    for check in task_of(messages)["dataset"]["failed_checks"]:
        if check["rule_id"] == rule_id and check["column"] == column:
            return check["evidence_id"]
    raise AssertionError(f"No seeded evidence for {rule_id}/{column}")


def submit_with(builder: Callable[[list[BaseMessage]], dict[str, Any]]) -> Callable[[list[BaseMessage]], AIMessage]:
    """A response that builds its submission from the conversation so far."""
    return lambda messages: submit(builder(messages))
