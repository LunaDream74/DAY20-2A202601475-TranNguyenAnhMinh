"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    system_provider: Literal["openai", "openrouter"] = Field(
        default="openai", validation_alias="SYSTEM_PROVIDER"
    )
    system_model: str | None = Field(default=None, validation_alias="SYSTEM_MODEL")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    eval_judge_provider: Literal["openai"] = Field(
        default="openai", validation_alias="EVAL_JUDGE_PROVIDER"
    )
    eval_judge_model: str | None = Field(default=None, validation_alias="EVAL_JUDGE_MODEL")
    openai_input_cost_per_million: float | None = Field(
        default=None, ge=0, validation_alias="OPENAI_INPUT_COST_PER_MILLION"
    )
    openai_output_cost_per_million: float | None = Field(
        default=None, ge=0, validation_alias="OPENAI_OUTPUT_COST_PER_MILLION"
    )
    openai_max_retries: int = Field(default=1, ge=0, le=5, validation_alias="OPENAI_MAX_RETRIES")
    openai_request_timeout_seconds: int = Field(
        default=30, ge=5, le=120, validation_alias="OPENAI_REQUEST_TIMEOUT_SECONDS"
    )

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
