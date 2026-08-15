"""Cross-lingual semantic retrieval service."""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.embeddings.base import EmbeddingProvider
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.retrieval.routing import LanguageRoute, route_query
from multilingual_rag.transliteration.base import Transliterator
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
        transliterator: Transliterator | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.language_detector = language_detector or LanguageDetector()
        self.transliterator = transliterator

    def route(
        self,
        query: str,
        *,
        force_language: str | None = None,
        skip_transliteration: bool = False,
    ) -> LanguageRoute:
        """Decide which text to embed for ``query`` (the transliteration decision).

        Public so the agent graph can make this decision as an explicit step, and re-make it
        differently on a weak retrieval. ``retrieve`` calls it for callers that don't.
        """
        return route_query(
            query.strip(),
            settings=self.settings,
            transliterator=self.transliterator,
            force_language=force_language,
            skip_transliteration=skip_transliteration,
        )

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
        route: LanguageRoute | None = None,
    ) -> RetrievalContext:
        """Retrieve context chunks for a query, scoped to one user (and chat, when given).

        When the query is detected as **romanized Indic** and transliteration is enabled, the
        query is transliterated to its native script and *that* form is embedded and searched —
        so it matches the native-script index instead of collapsing to noise. A plain English
        query has no such markers, so it is searched as-is and stays same-language. Detection (a
        linguistic check) is used rather than routing by retrieval score, which proved unreliable
        at scale.

        Pass ``route`` to reuse a decision already made (the agent graph does this, so a repaired
        retry can override the script choice without re-detecting).
        """
        normalized_query = query.strip()
        decided = route if route is not None else self.route(normalized_query)
        query_language = self.language_detector.detect(normalized_query)
        limit = top_k or self.settings.retrieval_top_k

        embedding = self.embedding_provider.embed_query(decided.search_text)
        results = self.vector_store.search(
            embedding, user_id=user_id, session_id=session_id, top_k=limit, filters=filters
        )

        return RetrievalContext(
            query=normalized_query,
            query_language=query_language,
            results=results,
            transliterated_query=decided.transliterated_query,
            transliteration_applied=decided.transliteration_applied,
        )
