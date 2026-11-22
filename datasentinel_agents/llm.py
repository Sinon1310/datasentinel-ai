"""Groq chat-model factory and transient-error classification."""

from __future__ import annotations

import logging
from typing import Callable

import groq
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from datasentinel.config import Settings, get_settings
from datasentinel.exceptions import DataSentinelError
from datasentinel_agents.config import AgentSettings, get_agent_settings

logger = logging.getLogger(__name__)

LLMFactory = Callable[[str], BaseChatModel]
"""Callable returning a chat model for a given agent name."""


class LLMConfigurationError(DataSentinelError):
    """The LLM provider is not configured (e.g. missing API key)."""


class TransientLLMError(Exception):
    """Provider-agnostic transient failure (used by tests and adapters)."""


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
    TransientLLMError,
)


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying (rate limits, timeouts, 5xx, connection problems)."""
    return isinstance(exc, TRANSIENT_ERRORS)


def groq_llm_factory(settings: Settings | None = None, agent_settings: AgentSettings | None = None) -> LLMFactory:
    """Build a factory of ``ChatGroq`` models.

    ``max_retries=0`` so the LangGraph ``RetryPolicy`` is the single retry layer.

    Raises:
        LLMConfigurationError: If ``GROQ_API_KEY`` is not configured.
    """
    settings = settings or get_settings()
    agent_settings = agent_settings or get_agent_settings()
    key = settings.groq_api_key
    if key is None or not key.get_secret_value().strip():
        raise LLMConfigurationError("GROQ_API_KEY is not set. Add it to your environment or .env file.")
    model_name = agent_settings.model or settings.groq_model

    def factory(agent_name: str) -> BaseChatModel:
        logger.debug("Creating ChatGroq(%s) for %s", model_name, agent_name)
        return ChatGroq(
            model=model_name,
            temperature=agent_settings.temperature,
            timeout=agent_settings.request_timeout_seconds,
            max_retries=0,
            api_key=key,
        )

    return factory
