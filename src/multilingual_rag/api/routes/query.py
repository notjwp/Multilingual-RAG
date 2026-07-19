"""Query route for retrieval-augmented generation."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    AnswerCitation,
    GeneratedAnswer,
    RetrievalContext,
    UserRecord,
    VectorSearchResult,
)
from multilingual_rag.embeddings.factory import build_embedding_provider
from multilingual_rag.generation.base import AnswerGenerator
from multilingual_rag.generation.openai_compatible_generator import (
    OpenAICompatibleAnswerGenerator,
)
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.vectorstores.base import MetadataValue, VectorFilter
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

router = APIRouter(prefix="/v1", tags=["query"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)


class QueryRequest(BaseModel):
    """Request body for a RAG query."""

    query: str = Field(min_length=1)
    preferred_language: str | None = Field(default=None, min_length=2)
    filters: dict[str, MetadataValue] | None = None
    top_k: int | None = Field(default=None, gt=0, le=50)


class CitationResponse(BaseModel):
    """Citation returned with an answer."""

    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    text: str


class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk returned for transparency/debugging."""

    chunk_id: str
    document_id: str
    text: str
    language: str
    source: str
    chunk_index: int
    score: float
    page: int | None = None
    token_count: int
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Response body for a RAG query."""

    answer: str
    language: str
    query_language: str
    citations: tuple[CitationResponse, ...]
    retrieved_chunks: tuple[RetrievedChunkResponse, ...]


class QueryService(Protocol):
    """Protocol for API query orchestration."""

    def answer_query(self, request: QueryRequest, *, user_id: str) -> QueryResponse:
        """Answer one query request for a user."""
        ...


class RagQueryService:
    """Coordinate retrieval and generation for API queries."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator

    def answer_query(self, request: QueryRequest, *, user_id: str) -> QueryResponse:
        """Retrieve context and generate an answer for a user's query."""
        if request.filters and "user_id" in request.filters:
            raise AppError(
                "user_id is not an allowed filter.",
                code="reserved_filter_key",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        context = self.retrieval_service.retrieve(
            request.query,
            user_id=user_id,
            top_k=request.top_k,
            filters=cast(VectorFilter | None, request.filters),
        )
        generated_answer = self.answer_generator.generate_answer(
            context=context,
            preferred_language=request.preferred_language,
        )
        return query_response_from_models(generated_answer, context)


@router.post("/query", response_model=QueryResponse)
async def query(
    request_body: QueryRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
) -> QueryResponse:
    """Answer a user query with retrieval-augmented generation."""
    query_service = get_query_service(request)
    # The RAG core is synchronous and blocking (local bge-m3 embedding, Chroma search, a
    # generation HTTP call) — offload it so it doesn't stall the event loop for other requests.
    return await asyncio.to_thread(
        query_service.answer_query, request_body, user_id=current_user.user_id
    )


def get_query_service(request: Request) -> QueryService:
    """Return an injected or default query service, built once and memoized on app.state."""
    existing_service = getattr(request.app.state, "query_service", None)
    if existing_service is not None:
        return cast(QueryService, existing_service)

    settings = cast(Settings, request.app.state.settings)
    vector_store = ChromaVectorStore(settings)
    retrieval_service = RetrievalService(
        settings,
        embedding_provider=build_embedding_provider(settings),
        vector_store=vector_store,
    )
    service = RagQueryService(
        retrieval_service=retrieval_service,
        answer_generator=OpenAICompatibleAnswerGenerator(settings),
    )
    # Cache so the Chroma client and adapters aren't rebuilt on every request. Lazy (not in the
    # lifespan) so the offline test suite never loads the 2.2 GB model at startup.
    request.app.state.query_service = service
    return service


def query_response_from_models(answer: GeneratedAnswer, context: RetrievalContext) -> QueryResponse:
    """Map domain models to the public API response."""
    return QueryResponse(
        answer=answer.answer,
        language=answer.language,
        query_language=context.query_language,
        citations=tuple(citation_response(citation) for citation in answer.citations),
        retrieved_chunks=tuple(chunk_response(result) for result in context.results),
    )


def citation_response(citation: AnswerCitation) -> CitationResponse:
    return CitationResponse(**citation.model_dump())


def chunk_response(result: VectorSearchResult) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(**result.model_dump())
