"""Vector store contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from multilingual_rag.core.models import DocumentChunk, VectorSearchResult
from multilingual_rag.embeddings.base import EmbeddingVector

MetadataValue = str | int | float | bool
VectorFilter = Mapping[str, MetadataValue]


class VectorStore(Protocol):
    """Protocol for vector database integrations."""

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> None:
        """Insert or update chunk embeddings, scoped to one user (and chat, when given)."""
        ...

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search one user's (and chat's, when given) chunks for relevant matches."""
        ...

    def delete_document(
        self, document_id: str, *, user_id: str, session_id: str | None = None
    ) -> None:
        """Delete all of one user's (and chat's, when given) chunks for a document."""
        ...

