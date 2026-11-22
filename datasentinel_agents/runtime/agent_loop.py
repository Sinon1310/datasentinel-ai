"""The tool-calling loop shared by all agents.

Protocol: the model may call its read-only tools, and must finish by calling
``submit_report`` whose argument schema *is* the agent's Pydantic output model. That gives
structured output that works with tool-calling Llama models on Groq, and lets the same
loop validate and repair the submission:

* invalid submission  → Pydantic errors returned to the model; up to ``max_repair_attempts``
* no tool call        → reminder to submit; counts as a repair
* tool budget reached → only ``submit_report`` remains available (forced tool choice)

Transient provider errors are *not* caught here; they propagate so LangGraph's
``RetryPolicy`` can retry the node. The message history is private to this run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from datasentinel_agents.config import AgentSettings
from datasentinel_agents.models import SUBMISSION_CONTEXT, AgentError, AgentRunStatus, Evidence, TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
SUBMIT_TOOL = "submit_report"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentOutcome(Generic[T]):
    """Result of one agent run."""

    agent: str
    output: T | None
    status: AgentRunStatus
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[AgentError] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


def _submit_tool(output_model: type[BaseModel]) -> StructuredTool:
    def _submit(**_: Any) -> str:  # never executed; submissions are validated by the loop
        return "received"

    return StructuredTool.from_function(
        func=_submit,
        name=SUBMIT_TOOL,
        description=f"Submit your final {output_model.__name__}. Call exactly once when your analysis is complete.",
        args_schema=output_model,
    )


def _error_details(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in exc.errors(include_url=False)[:12]]


class AgentRunner:
    """Run one agent to a validated structured output."""

    def __init__(self, llm: BaseChatModel, settings: AgentSettings) -> None:
        self.llm = llm
        self.settings = settings

    def run(
        self,
        agent: str,
        system_prompt: str,
        task: dict[str, Any],
        tools: list[BaseTool],
        output_model: type[T],
        recorder_records: list[Evidence],
    ) -> AgentOutcome[T]:
        """Execute the loop. ``recorder_records`` is the live list the tools append evidence to."""
        submit = _submit_tool(output_model)
        tool_map = {t.name: t for t in tools}
        model = self.llm.bind_tools([*tools, submit])
        forced = self.llm.bind_tools([submit], tool_choice=SUBMIT_TOOL)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Task (JSON):\n" + json.dumps(task, default=str)),
        ]
        usage = TokenUsage()
        errors: list[AgentError] = []
        tool_calls = repairs = 0
        max_steps = self.settings.max_tool_calls + self.settings.max_repair_attempts + 3

        for _ in range(max_steps):
            budget_spent = tool_calls >= self.settings.max_tool_calls
            response = (forced if budget_spent else model).invoke(messages)
            self._track_usage(usage, response)
            messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                repairs += 1
                if repairs > self.settings.max_repair_attempts:
                    break
                messages.append(HumanMessage(content=f"You must finish by calling `{SUBMIT_TOOL}` with your structured report."))
                continue

            submitted: T | None = None
            for call in calls:
                name, call_id, args = call.get("name"), call.get("id") or "call", call.get("args") or {}
                if name == SUBMIT_TOOL:
                    try:
                        submitted = output_model.model_validate(args, context=SUBMISSION_CONTEXT)
                        messages.append(ToolMessage(content="accepted", tool_call_id=call_id))
                    except ValidationError as exc:
                        repairs += 1
                        errors.append(AgentError(agent=agent, kind="validation", timestamp=_now(),
                                                 message=f"Invalid submission: {_error_details(exc)[:3]}"))
                        messages.append(ToolMessage(
                            content=json.dumps({"error": "validation_failed", "fix_these": _error_details(exc)}),
                            tool_call_id=call_id))
                elif name in tool_map:
                    tool_calls += 1
                    if tool_calls > self.settings.max_tool_calls:
                        content = json.dumps({"error": f"Tool budget exhausted. Call {SUBMIT_TOOL} now."})
                    else:
                        content = self._invoke_tool(tool_map[name], args)
                    messages.append(ToolMessage(content=content, tool_call_id=call_id))
                else:
                    messages.append(ToolMessage(content=json.dumps({"error": f"Unknown tool '{name}'."}), tool_call_id=call_id))

            if submitted is not None:
                status = AgentRunStatus(status="succeeded", tool_calls=tool_calls, repairs=repairs)
                logger.info("%s submitted after %d tool call(s), %d repair(s)", agent, tool_calls, repairs)
                return AgentOutcome(agent, submitted, status, list(recorder_records), errors, usage)
            if repairs > self.settings.max_repair_attempts:
                break

        message = f"No valid {output_model.__name__} after {repairs} repair attempt(s) and {tool_calls} tool call(s)."
        logger.warning("%s degraded: %s", agent, message)
        errors.append(AgentError(agent=agent, kind="validation", message=message, timestamp=_now()))
        status = AgentRunStatus(status="degraded", tool_calls=tool_calls, repairs=repairs, message=message)
        return AgentOutcome(agent, None, status, list(recorder_records), errors, usage)

    @staticmethod
    def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> str:
        try:
            result = tool.invoke(args)
        except Exception as exc:  # noqa: BLE001 — argument errors go back to the model
            return json.dumps({"error": f"Invalid arguments: {str(exc)[:300]}"})
        return result if isinstance(result, str) else json.dumps(result, default=str)

    @staticmethod
    def _track_usage(usage: TokenUsage, response: AIMessage) -> None:
        usage.calls += 1
        meta = getattr(response, "usage_metadata", None) or {}
        usage.input_tokens += int(meta.get("input_tokens", 0) or 0)
        usage.output_tokens += int(meta.get("output_tokens", 0) or 0)
