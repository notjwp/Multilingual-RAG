"""Select the embedding provider from settings.

bge-m3 (local, free) is the default per M0; OpenAI stays available behind config. Both satisfy
the ``EmbeddingProvider`` protocol, so nothing downstream changes.
"""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.embeddings.base import EmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Return the configured embedding provider."""
    if settings.embedding_provider == "openai":
        from multilingual_rag.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(settings)

    from multilingual_rag.embeddings.bge_embeddings import BgeM3EmbeddingProvider

    return BgeM3EmbeddingProvider(device=settings.embedding_device)
