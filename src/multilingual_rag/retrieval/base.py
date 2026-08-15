"""Retrieval contract (a port, like ``EmbeddingProvider`` and ``VectorStore``)."""

from __future__ import annotations

from typing import Protocol

from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.retrieval.routing import LanguageRoute
from multilingual_rag.vectorstores.base import VectorFilter


class Retriever(Protocol):
    """Route a query to a script, then search one user's (and chat's) chunks.

    The agent graph depends on this rather than on ``RetrievalService`` directly, so its nodes can
    be exercised with a plain fake.
    """

    def route(
        self,
        query: str,
        *,
        force_language: str | None = None,
        skip_transliteration: bool = False,
    ) -> LanguageRoute:
        """Decide which text to embed for ``query``."""
        ...

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
        """Retrieve context chunks. Decides the route itself when one isn't supplied."""
        ...
