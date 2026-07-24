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

    # Browser origins allowed to call the API (the M16 frontend + local dev). Accepts a
    # comma-separated env string (CORS_ALLOW_ORIGINS=http://localhost:3000,https://app.example).
    cors_allow_origins: tuple[str, ...] = ("http://localhost:3000",)

    # Embedding provider: bge-m3 (local, free) is the default; openai stays available.
    embedding_provider: Literal["bge-m3", "openai"] = "bge-m3"
    embedding_device: str | None = None  # bge-m3 torch device; None = auto-select CUDA
    # Load the local embedding model at API startup instead of lazily on the first query — moves
    # the ~10s cold-start to boot so the first message isn't slow. Off by default so the test suite
    # and offline tooling never load the 2.2 GB model at startup.
    warm_embeddings_on_startup: bool = False
    # When the embedding model is already cached locally, skip Hugging Face Hub network checks on
    # load (~6s of round-trips per process). Enable only if the model is fully cached; leave off
    # for first-time setup so the weights can still be downloaded.
    hf_hub_offline: bool = False

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
    # How to decide a query is romanized Indic (whether/what to transliterate). "word-list"
    # (default) is Hindi-only. "muril" is a local multi-class classifier (hi/kn/te, ~950 MB model,
    # no network). "google" detects the language via Google Translate (hi/kn/te, a network call per
    # query). muril or google + transliteration_languages=hi,kn,te enable Kannada/Telugu. See
    # docs/architecture.md §1.5b.
    transliteration_detector: Literal["word-list", "muril", "google"] = "word-list"

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

    # Multi-turn chat: how many prior messages to feed the answer model (and the query-rewrite
    # step) as conversation history. ~5 exchanges. 0 disables history (single-shot per turn).
    chat_history_max_messages: int = Field(default=10, ge=0)

    @field_validator("transliteration_languages", "cors_allow_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """Accept a comma-separated env string (``hi,kn,te``), not just JSON, for convenience."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

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
