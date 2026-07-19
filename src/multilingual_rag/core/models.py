"""Shared domain models for document ingestion and retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata describing a source document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source: str
    content_type: str
    checksum: str
    language: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A logical section extracted from a document."""

    model_config = ConfigDict(frozen=True)

    text: str
    page: int | None = None
    section_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """A parsed document before chunking."""

    model_config = ConfigDict(frozen=True)

    source_path: Path
    content_type: str
    sections: tuple[DocumentSection, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of document text ready for embedding."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    language: str
    source: str
    chunk_index: int
    checksum: str
    page: int | None = None
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result returned after parsing, language detection, and chunking."""

    model_config = ConfigDict(frozen=True)

    document: DocumentMetadata
    chunks: tuple[DocumentChunk, ...]


class VectorSearchResult(BaseModel):
    """A document chunk returned from vector search."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    language: str
    source: str
    chunk_index: int
    score: float
    page: int | None = None
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """Retrieved context for one user query."""

    model_config = ConfigDict(frozen=True)

    query: str
    query_language: str
    results: tuple[VectorSearchResult, ...]
    # Set when a romanized query was transliterated to native script and dual-queried, so callers
    # can see the form that was actually searched. None/False for the ordinary single-query path.
    transliterated_query: str | None = None
    transliteration_applied: bool = False


class AnswerCitation(BaseModel):
    """Citation for a generated answer."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    text: str


class GeneratedAnswer(BaseModel):
    """Generated answer and citations."""

    model_config = ConfigDict(frozen=True)

    answer: str
    language: str
    citations: tuple[AnswerCitation, ...]


class DocumentRecord(BaseModel):
    """Persisted metadata for an indexed document."""

    model_config = ConfigDict(frozen=True)

    document: DocumentMetadata
    chunk_count: int = Field(ge=0)


class UserRecord(BaseModel):
    """Authenticated user identity."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    email: str


class IngestionJobRecord(BaseModel):
    """Persisted ingestion job state."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    user_id: str
    file_path: str
    status: str
    document_id: str | None = None
    error_message: str | None = None
