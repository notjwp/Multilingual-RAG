from pathlib import Path

import pytest
from pydantic import ValidationError

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

