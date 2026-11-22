"""Agent-layer configuration (env prefix ``DATASENTINEL_AGENT_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Tunable limits for the agent layer. Secrets stay in :class:`datasentinel.config.Settings`."""

    model_config = SettingsConfigDict(env_prefix="DATASENTINEL_AGENT_", env_file=".env", extra="ignore")

    model: str | None = Field(default=None, description="Per-deployment override; defaults to Settings.groq_model (GROQ_MODEL).")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    max_tool_calls: int = Field(default=8, ge=1, le=30, description="Per-agent tool-call budget.")
    max_repair_attempts: int = Field(default=2, ge=0, le=5, description="Structured-output repair attempts.")
    llm_retry_attempts: int = Field(default=3, ge=1, le=6, description="Node retries on transient Groq errors.")
    retry_initial_interval: float = Field(default=2.0, ge=0.0)

    max_followup_rounds: int = Field(default=1, ge=0, le=2)
    max_remediation_repairs: int = Field(default=1, ge=0, le=2)
    max_fixes: int = Field(default=6, ge=1, le=10)
    dry_run_rows: int = Field(default=5000, ge=100)
    sample_row_limit: int = Field(default=10, ge=1, le=20)
    recursion_limit: int = Field(default=40, ge=10)
    checkpoint_filename: str = "checkpoints.sqlite"


@lru_cache(maxsize=1)
def get_agent_settings() -> AgentSettings:
    """Cached agent settings."""
    return AgentSettings()
