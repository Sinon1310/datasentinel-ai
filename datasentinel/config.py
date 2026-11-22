"""Application configuration loaded from environment variables / ``.env``.

All runtime settings are read through :func:`get_settings`. Secrets such as the Groq API
key are *never* hardcoded; they are read from the environment and wrapped in
:class:`pydantic.SecretStr` so they are not accidentally logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class Settings(BaseSettings):
    """Runtime settings for DataSentinel AI.

    Every field can be overridden with an environment variable prefixed with
    ``DATASENTINEL_`` (e.g. ``DATASENTINEL_MAX_UPLOAD_SIZE_MB=50``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DATASENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DataSentinel AI"
    log_level: str = "INFO"
    max_upload_size_mb: float = Field(default=200.0, gt=0, description="Maximum accepted upload size.")
    allowed_extensions: Tuple[str, ...] = (".csv", ".json", ".jsonl", ".ndjson")

    # LLM provider (Milestone 2). Read from GROQ_API_KEY or DATASENTINEL_GROQ_API_KEY.
    groq_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "DATASENTINEL_GROQ_API_KEY"),
    )
    # Must support tool calling. GROQ_MODEL takes precedence over the legacy DATASENTINEL_GROQ_MODEL.
    groq_model: str = Field(
        default=DEFAULT_GROQ_MODEL,
        validation_alias=AliasChoices("GROQ_MODEL", "DATASENTINEL_GROQ_MODEL"),
    )

    # Local persistence for investigation runs (datasets, checkpoints, audit logs).
    data_dir: Path = Path(".datasentinel")

    @field_validator("groq_model", mode="before")
    @classmethod
    def _blank_model_uses_default(cls, value: object) -> object:
        """Treat ``GROQ_MODEL=`` (blank) as unset rather than sending an empty model name."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_GROQ_MODEL
        return value.strip() if isinstance(value, str) else value

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return int(self.max_upload_size_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
