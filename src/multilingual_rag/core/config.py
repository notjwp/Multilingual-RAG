"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]

DEFAULT_JWT_SECRET = "change-me-in-production"


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

    # Embedding provider: bge-m3 (local, free) is the default; openai stays available.
    embedding_provider: Literal["bge-m3", "openai"] = "bge-m3"
    embedding_device: str | None = None  # bge-m3 torch device; None = auto-select CUDA

    # Generation: any OpenAI-compatible chat-completions endpoint. The provider is a URL, not a
    # code path — NVIDIA NIM (default), OpenRouter, Groq, a local Ollama/vLLM shim, or OpenAI
    # itself are all reachable by changing generation_base_url alone.
    generation_base_url: str = "https://integrate.api.nvidia.com/v1"
    generation_api_key: SecretStr | None = None
    # Catalogs rotate — verify the id at your provider (e.g. build.nvidia.com/models).
    generation_model: str = "meta/llama-3.1-8b-instruct"
    # Fail fast instead of blocking on a cold/overloaded model (SDK default is 600s).
    generation_timeout_seconds: float = Field(default=60.0, gt=0)

    # Romanized-Indic query support: detect romanized Hindi and transliterate it to native script
    # before embedding, so it matches the native-script index (plain English is left untouched).
    # Default provider is google (googletrans; best quality, free, but a network call per query)
    # with a local rule-based fallback; indicxlit is the offline neural option.
    transliteration_enabled: bool = True
    transliteration_provider: Literal["google", "indicxlit", "rule-based", "llm", "off"] = (
        "google"
    )
    transliteration_languages: tuple[str, ...] = ("hi",)
    # How to decide a query is romanized Hindi (whether to transliterate). "word-list" (default) is
    # a fast local function-word check; "muril" opts into a MuRIL feature + LR classifier (downloads
    # a ~950 MB model, adds a forward pass per Latin query — no measured accuracy gain over the word
    # list on the eval, but generalizes better to real-world spelling variants).
    transliteration_detector: Literal["word-list", "muril"] = "word-list"

    # Only used when embedding_provider is "openai".
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_batch_size: int = Field(default=96, gt=0)

    chroma_persist_directory: Path = Path("data/chroma")
    chroma_collection_name: str = "multilingual_documents"
    raw_document_directory: Path = Path("data/raw")
    max_upload_bytes: int = Field(default=25_000_000, gt=0)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/multilingual_rag"
    jwt_secret_key: SecretStr = SecretStr(DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60, gt=0)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

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

    @model_validator(mode="after")
    def reject_default_secret_in_prod(self) -> Self:
        """Refuse to boot deployed environments with the placeholder JWT secret."""
        if (
            self.environment in ("production", "staging")
            and self.jwt_secret_key.get_secret_value() == DEFAULT_JWT_SECRET
        ):
            raise ValueError(
                "jwt_secret_key must be set to a non-default value in "
                f"{self.environment}; refusing to start with the placeholder secret."
            )
        return self

    @model_validator(mode="after")
    def require_generation_credentials_in_prod(self) -> Self:
        """Refuse to boot deployed environments without a generation API key.

        Only enforced for production/staging: local and test construct Settings without keys,
        and the generator raises a clear AppError naming the missing variable if it is ever
        actually used.
        """
        if self.environment not in ("production", "staging"):
            return self
        if self.generation_api_key is None:
            raise ValueError(
                "generation_api_key is required; set GENERATION_API_KEY in "
                f"{self.environment} (see GENERATION_BASE_URL for the provider)."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
