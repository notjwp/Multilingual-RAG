"""Cross-lingual semantic retrieval service."""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.embeddings.base import EmbeddingProvider
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.vectorstores.base import VectorFilter, VectorStore


class RetrievalService:
    """Embed user queries and retrieve semantically relevant chunks."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.language_detector = language_detector or LanguageDetector()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
    ) -> RetrievalContext:
        """Retrieve one user's context chunks for a query."""
        normalized_query = query.strip()
        query_language = self.language_detector.detect(normalized_query)
        query_embedding = self.embedding_provider.embed_query(normalized_query)
        results = self.vector_store.search(
            query_embedding,
            user_id=user_id,
            top_k=top_k or self.settings.retrieval_top_k,
            filters=filters,
        )
        return RetrievalContext(
            query=normalized_query,
            query_language=query_language,
            results=results,
        )

