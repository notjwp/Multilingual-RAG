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
    ) -> None:
        """Insert or update chunk embeddings."""
        ...

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search the vector store for relevant chunks."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        ...

