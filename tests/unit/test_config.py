from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from multilingual_rag.core.config import Settings


def test_settings_defaults_are_valid() -> None:
    settings = Settings()

    assert settings.app_name == "multilingual-rag"
    assert settings.environment == "local"
    assert settings.api_prefix == "/v1"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.chroma_persist_directory == Path("data/chroma")


def test_api_prefix_strips_trailing_slash() -> None:
    settings = Settings(api_prefix="/api/")

    assert settings.api_prefix == "/api"


def test_api_prefix_must_start_with_slash() -> None:
    with pytest.raises(ValidationError, match="api_prefix must start"):
        Settings(api_prefix="v1")


def test_log_level_is_normalized() -> None:
    settings = Settings(log_level="debug")

    assert settings.log_level == "DEBUG"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_tokens must be smaller"):
        Settings(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_jwt_default_secret_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="placeholder secret"):
        Settings(environment="production")


def test_short_jwt_secret_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings(environment="production", jwt_secret_key=SecretStr("too-short-secret"))


def test_strong_jwt_secret_accepted_in_production() -> None:
    settings = Settings(
        environment="production",
        jwt_secret_key=SecretStr("x" * 32),
        generation_api_key=SecretStr("test-key"),
    )
    assert settings.environment == "production"


def test_weak_jwt_secret_allowed_outside_prod() -> None:
    # The strength rule only applies to production/staging.
    settings = Settings(environment="local", jwt_secret_key=SecretStr("short"))
    assert settings.jwt_secret_key.get_secret_value() == "short"

