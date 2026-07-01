"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]


class Settings(BaseSettings):
    """Runtime settings for the multilingual RAG service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "multilingual-rag"
    environment: Environment = "local"
    log_level: str = "INFO"
    api_prefix: str = "/v1"

    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_generation_model: str = "gpt-4.1-mini"

    chroma_persist_directory: Path = Path("data/chroma")
    chroma_collection_name: str = "multilingual_documents"

    chunk_size_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=8, gt=0)

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Ensure API prefixes are absolute paths without a trailing slash."""
        if not value.startswith("/"):
            raise ValueError("api_prefix must start with '/'")
        if len(value) > 1 and value.endswith("/"):
            return value.rstrip("/")
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize logging levels for consistent configuration."""
        normalized = value.upper()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in valid_levels:
            raise ValueError(f"log_level must be one of: {', '.join(sorted(valid_levels))}")
        return normalized

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: object) -> int:
        """Keep overlap smaller than chunk size when both values are available."""
        data = getattr(info, "data", {})
        chunk_size_tokens = data.get("chunk_size_tokens")
        if chunk_size_tokens is not None and value >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
