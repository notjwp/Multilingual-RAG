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
    ) -> None:
        """Insert or update chunk embeddings for one user."""
        ...

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search one user's chunks for relevant matches."""
        ...

    def delete_document(self, document_id: str, *, user_id: str) -> None:
        """Delete all of one user's chunks for a document."""
        ...

