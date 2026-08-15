"""Shared route dependencies.

The agent graph lives here rather than in ``routes/query.py`` because all three orchestrations
(``/v1/query``, blocking chat, streaming chat) now share it — it is no longer a query-route
concept. Follows the repo's injection convention: return the ``app.state`` attr when a test has
set one, otherwise build and memoize.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from multilingual_rag.agent.factory import build_rag_graph
from multilingual_rag.agent.graph import RagGraph
from multilingual_rag.core.config import Settings


def get_rag_graph(request: Request) -> RagGraph:
    """Return an injected or default agent graph, built once and memoized on app.state.

    Building it constructs the embedding provider, vector store, transliterator, stream client and
    grader exactly once per process, so the 2.2 GB model has a single owner. Lazy (not in the
    lifespan) so the offline test suite never loads it at startup.
    """
    existing = getattr(request.app.state, "rag_graph", None)
    if existing is not None:
        return cast(RagGraph, existing)

    graph = build_rag_graph(cast(Settings, request.app.state.settings))
    request.app.state.rag_graph = graph
    return graph
