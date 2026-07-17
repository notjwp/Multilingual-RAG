"""The embedding factory selects the provider from settings — without loading any model.

BgeM3EmbeddingProvider defers its 2.2 GB load to first use (module-level lru_cache), so simply
constructing it here is cheap and offline.
"""

from pydantic import SecretStr

from multilingual_rag.core.config import Settings
from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider
from multilingual_rag.embeddings.factory import build_embedding_provider
from multilingual_rag.embeddings.openai_embeddings import OpenAIEmbeddingProvider


def test_factory_defaults_to_bge_m3() -> None:
    provider = build_embedding_provider(Settings(environment="test"))
    assert isinstance(provider, BgeM3EmbeddingProvider)


def test_factory_passes_device_to_bge_m3() -> None:
    provider = build_embedding_provider(Settings(environment="test", embedding_device="cpu"))
    assert isinstance(provider, BgeM3EmbeddingProvider)
    assert provider.device == "cpu"


def test_factory_returns_openai_when_configured() -> None:
    settings = Settings(
        environment="test",
        embedding_provider="openai",
        openai_api_key=SecretStr("test-key"),
    )
    assert isinstance(build_embedding_provider(settings), OpenAIEmbeddingProvider)
