"""Cross-lingual semantic retrieval service."""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.embeddings.base import EmbeddingProvider
from multilingual_rag.ingestion.language import LanguageDetector
from multilingual_rag.transliteration.base import Transliterator
from multilingual_rag.transliteration.detect import detect_target_language
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

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
    ) -> RetrievalContext:
        """Retrieve one user's context chunks for a query.

        When the query is detected as **romanized Hindi** (`is_romanized_indic`) and
        transliteration is enabled, the query is transliterated to native Devanagari and *that*
        form is embedded and searched — so it matches the native-script index instead of
        collapsing to noise. A plain English query has no Hindi markers, so it is searched as-is
        and stays same-language. Detection (a linguistic check) is used rather than routing by
        retrieval score, which proved unreliable at scale.
        """
        normalized_query = query.strip()
        query_language = self.language_detector.detect(normalized_query)
        limit = top_k or self.settings.retrieval_top_k

        transliterated_query = self._transliterate(normalized_query)
        search_text = transliterated_query if transliterated_query is not None else normalized_query
        embedding = self.embedding_provider.embed_query(search_text)
        results = self.vector_store.search(
            embedding, user_id=user_id, top_k=limit, filters=filters
        )

        return RetrievalContext(
            query=normalized_query,
            query_language=query_language,
            results=results,
            transliterated_query=transliterated_query,
            transliteration_applied=transliterated_query is not None,
        )

    def _transliterate(self, query: str) -> str | None:
        """Return the native-script transliteration to search with, or None to leave the query.

        Detects *which* configured Indic language the query is romanized in and transliterates to
        that script (Hindi with the default detector; hi/kn/te with the ``google`` detector).
        Skips when nothing is detected or the transliterator returns the input unchanged (a no-op).
        """
        if self.transliterator is None:
            return None
        target = detect_target_language(
            query,
            self.settings.transliteration_languages,
            detector=self.settings.transliteration_detector,
        )
        if target is None:
            return None
        transliterated = self.transliterator.transliterate(query, target_language=target)
        if not transliterated.strip() or transliterated.strip() == query.strip():
            return None
        return transliterated
