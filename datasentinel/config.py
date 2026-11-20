"""Application configuration loaded from environment variables / ``.env``.

All runtime settings are read through :func:`get_settings`. Secrets such as the Groq API
key are *never* hardcoded; they are read from the environment and wrapped in
:class:`pydantic.SecretStr` so they are not accidentally logged.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Reserved for Milestone 2 (LLM reasoning). Read from GROQ_API_KEY or DATASENTINEL_GROQ_API_KEY.
    groq_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "DATASENTINEL_GROQ_API_KEY"),
    )
    groq_model: str = "llama-3.3-70b-versatile"

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return int(self.max_upload_size_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
