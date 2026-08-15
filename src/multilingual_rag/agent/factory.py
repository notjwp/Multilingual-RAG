"""Build the agent graph from settings (the ``build_*`` factory convention).

This is the single place the whole RAG stack gets constructed — embedding provider, vector store,
transliterator, retriever, stream client, grader — so the expensive pieces exist once per process.
That removes by construction the hazard the old ``chat_stream.py`` patched with a
``cast(RagQueryService, ...).retrieval_service`` reach-through: there is no second owner of the
2.2 GB embedding model to keep in sync.
"""

from __future__ import annotations

from fastapi import status

from multilingual_rag.agent.grading.base import RelevanceGrader
from multilingual_rag.agent.grading.factory import build_relevance_grader
from multilingual_rag.agent.graph import RagGraph, build_graph
from multilingual_rag.agent.nodes import RagNodes
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.embeddings.factory import build_embedding_provider
from multilingual_rag.generation.base import StreamClient
from multilingual_rag.generation.openai_compatible_generator import OpenAICompatibleStreamClient
from multilingual_rag.retrieval.base import Retriever
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.transliteration.factory import build_transliterator
from multilingual_rag.vectorstores.factory import build_vector_store


def build_stream_client(settings: Settings) -> StreamClient:
    """Build the OpenAI-compatible streaming client, or fail with an actionable error."""
    api_key = settings.generation_api_key
    if api_key is None:
        raise AppError(
            "GENERATION_API_KEY is required to generate answers.",
            code="missing_generation_api_key",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return OpenAICompatibleStreamClient(
        api_key.get_secret_value(),
        settings.generation_base_url,
        settings.generation_timeout_seconds,
    )


def build_retriever(settings: Settings) -> Retriever:
    """Build the retrieval stack: embeddings + vector store + transliteration."""
    return RetrievalService(
        settings,
        embedding_provider=build_embedding_provider(settings),
        vector_store=build_vector_store(settings),
        transliterator=build_transliterator(settings),
    )


def build_rag_graph(
    settings: Settings,
    *,
    retriever: Retriever | None = None,
    client: StreamClient | None = None,
    grader: RelevanceGrader | None = None,
) -> RagGraph:
    """Assemble the agent graph. Every dependency is overridable for tests."""
    resolved_client = client if client is not None else build_stream_client(settings)
    nodes = RagNodes(
        settings,
        retriever=retriever if retriever is not None else build_retriever(settings),
        client=resolved_client,
        grader=grader
        if grader is not None
        else build_relevance_grader(settings, client=resolved_client),
    )
    return RagGraph(build_graph(nodes), max_repairs=settings.agent_max_repairs)
