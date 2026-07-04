"""Query route for retrieval-augmented generation."""

from __future__ import annotations

from typing import Any, Protocol, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    AnswerCitation,
    GeneratedAnswer,
    RetrievalContext,
    VectorSearchResult,
)
from multilingual_rag.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from multilingual_rag.generation.base import AnswerGenerator
from multilingual_rag.generation.openai_generator import OpenAIAnswerGenerator
from multilingual_rag.retrieval.service import RetrievalService
from multilingual_rag.vectorstores.base import MetadataValue, VectorFilter
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

router = APIRouter(prefix="/v1", tags=["query"])


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

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Answer one query request."""
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

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Retrieve context and generate an answer for a query."""
        context = self.retrieval_service.retrieve(
            request.query,
            top_k=request.top_k,
            filters=cast(VectorFilter | None, request.filters),
        )
        generated_answer = self.answer_generator.generate_answer(
            context=context,
            preferred_language=request.preferred_language,
        )
        return query_response_from_models(generated_answer, context)


@router.post("/query", response_model=QueryResponse)
async def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Answer a user query with retrieval-augmented generation."""
    query_service = get_query_service(request)
    return query_service.answer_query(request_body)


def get_query_service(request: Request) -> QueryService:
    """Return an injected or default query service."""
    existing_service = getattr(request.app.state, "query_service", None)
    if existing_service is not None:
        return cast(QueryService, existing_service)

    settings = cast(Settings, request.app.state.settings)
    embedding_provider = OpenAIEmbeddingProvider(settings)
    vector_store = ChromaVectorStore(settings)
    retrieval_service = RetrievalService(
        settings,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    return RagQueryService(
        retrieval_service=retrieval_service,
        answer_generator=OpenAIAnswerGenerator(settings),
    )


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
