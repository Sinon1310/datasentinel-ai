"""LangGraph wiring for the investigation workflow."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from datasentinel_agents.llm import is_transient_error
from datasentinel_agents.models import CHECKPOINT_MODELS
from datasentinel_agents.nodes.agents import (
    make_investigator_node,
    make_remediation_node,
    make_root_cause_node,
    make_schema_analyst_node,
)
from datasentinel_agents.nodes.control import (
    make_approval_node,
    make_execute_node,
    make_grounding_node,
    make_report_node,
    make_validate_fixes_node,
    make_verify_node,
)
from datasentinel_agents.nodes.prepare import make_prepare_node
from datasentinel_agents.nodes.routing import (
    make_route_after_root_cause,
    make_route_after_validation,
    route_after_approval,
    route_after_prepare,
)
from datasentinel_agents.services import Services
from datasentinel_agents.state import InvestigationState


def make_serializer() -> JsonPlusSerializer:
    """Checkpoint serializer with a strict allow-list of deserialisable model classes."""
    return JsonPlusSerializer(allowed_msgpack_modules=[(m.__module__, m.__name__) for m in CHECKPOINT_MODELS])


def build_graph(services: Services, checkpointer: BaseCheckpointSaver | None = None):
    """Compile the investigation graph.

    Flow::

        prepare_context → (data_investigator ‖ schema_analyst | report)
        → grounding_check → root_cause_analyst → (data_investigator [follow-up] | remediation_agent | report)
        remediation_agent → validate_fixes → (human_approval | remediation_agent [repair] | report)
        human_approval → (execute_fixes → verify | report) → report → END
    """
    settings = services.agent_settings
    retry = RetryPolicy(max_attempts=settings.llm_retry_attempts, initial_interval=settings.retry_initial_interval,
                        backoff_factor=2.0, max_interval=30.0, jitter=True, retry_on=is_transient_error)

    graph = StateGraph(InvestigationState)
    graph.add_node("prepare_context", make_prepare_node(services))
    graph.add_node("data_investigator", make_investigator_node(services), retry_policy=retry)
    graph.add_node("schema_analyst", make_schema_analyst_node(services), retry_policy=retry)
    graph.add_node("grounding_check", make_grounding_node(services))
    graph.add_node("root_cause_analyst", make_root_cause_node(services), retry_policy=retry)
    graph.add_node("remediation_agent", make_remediation_node(services), retry_policy=retry)
    graph.add_node("validate_fixes", make_validate_fixes_node(services))
    graph.add_node("human_approval", make_approval_node(services))
    graph.add_node("execute_fixes", make_execute_node(services))
    graph.add_node("verify", make_verify_node(services))
    graph.add_node("report", make_report_node(services))

    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges("prepare_context", route_after_prepare,
                                ["data_investigator", "schema_analyst", "report"])
    graph.add_edge("data_investigator", "grounding_check")
    graph.add_edge("schema_analyst", "grounding_check")
    graph.add_edge("grounding_check", "root_cause_analyst")
    graph.add_conditional_edges("root_cause_analyst", make_route_after_root_cause(settings),
                                ["data_investigator", "remediation_agent", "report"])
    graph.add_edge("remediation_agent", "validate_fixes")
    graph.add_conditional_edges("validate_fixes", make_route_after_validation(settings),
                                ["human_approval", "remediation_agent", "report"])
    graph.add_conditional_edges("human_approval", route_after_approval, ["execute_fixes", "report"])
    graph.add_edge("execute_fixes", "verify")
    graph.add_edge("verify", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
