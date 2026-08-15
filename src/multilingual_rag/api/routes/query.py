"""Query route for retrieval-augmented generation."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from multilingual_rag.api.dependencies import get_rag_graph
from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    AnswerCitation,
    GeneratedAnswer,
    RetrievalContext,
    UserRecord,
    VectorSearchResult,
)
from multilingual_rag.vectorstores.base import MetadataValue, VectorFilter

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
    # Present when a romanized query was transliterated and dual-queried (see RetrievalContext).
    transliterated_query: str | None = None
    transliteration_applied: bool = False


@router.post("/query", response_model=QueryResponse)
async def query(
    request_body: QueryRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
) -> QueryResponse:
    """Answer a user query with retrieval-augmented generation.

    No ``asyncio.to_thread`` here any more: the agent graph is async and offloads its own blocking
    calls (embedding, Chroma search, and language routing) inside the nodes that make them.
    """
    if request_body.filters and "user_id" in request_body.filters:
        # Tenancy is enforced server-side; a caller-supplied user_id filter is always a mistake.
        raise AppError(
            "user_id is not an allowed filter.",
            code="reserved_filter_key",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    result = await get_rag_graph(request).answer(
        request_body.query,
        user_id=current_user.user_id,
        preferred_language=request_body.preferred_language,
        top_k=request_body.top_k,
        filters=cast(VectorFilter | None, request_body.filters),
    )
    return query_response_from_models(result.answer, result.context)


def query_response_from_models(answer: GeneratedAnswer, context: RetrievalContext) -> QueryResponse:
    """Map domain models to the public API response."""
    return QueryResponse(
        answer=answer.answer,
        language=answer.language,
        query_language=context.query_language,
        citations=tuple(citation_response(citation) for citation in answer.citations),
        retrieved_chunks=tuple(chunk_response(result) for result in context.results),
        transliterated_query=context.transliterated_query,
        transliteration_applied=context.transliteration_applied,
    )


def citation_response(citation: AnswerCitation) -> CitationResponse:
    return CitationResponse(**citation.model_dump())


def chunk_response(result: VectorSearchResult) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(**result.model_dump())
