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

