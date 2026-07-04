"""ChromaDB vector store implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import chromadb
from fastapi import status

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentChunk, VectorSearchResult
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.vectorstores.base import MetadataValue, VectorFilter

ScalarMetadata = str | int | float | bool
type ChromaEmbedding = Sequence[float] | Sequence[int]


class ChromaVectorStore:
    """Persist and search document chunks in ChromaDB."""

    def __init__(self, settings: Settings) -> None:
        settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
    ) -> None:
        """Insert or update chunk embeddings in ChromaDB."""
        if not chunks:
            raise AppError(
                "At least one chunk is required for vector upsert.",
                code="empty_vector_upsert",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if len(chunks) != len(embeddings):
            raise AppError(
                "Chunk and embedding counts must match.",
                code="vector_upsert_size_mismatch",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        chroma_embeddings: list[ChromaEmbedding] = [embedding for embedding in embeddings]
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=chroma_embeddings,
            metadatas=[metadata_for_chunk(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search ChromaDB for chunks nearest to a query embedding."""
        if not query_embedding:
            raise AppError(
                "Query embedding must not be empty.",
                code="empty_query_embedding",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if top_k <= 0:
            raise AppError(
                "top_k must be greater than zero.",
                code="invalid_top_k",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        query_embeddings: list[ChromaEmbedding] = [query_embedding]
        query_result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=dict(filters) if filters else None,
            include=["documents", "metadatas", "distances"],
        )
        return parse_query_result(cast(dict[str, Any], query_result))

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        if not document_id.strip():
            raise AppError(
                "document_id is required.",
                code="missing_document_id",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        self._collection.delete(where={"document_id": document_id})


def metadata_for_chunk(chunk: DocumentChunk) -> dict[str, MetadataValue]:
    """Convert chunk metadata to Chroma-compatible scalar metadata."""
    metadata: dict[str, MetadataValue] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "language": chunk.language,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "checksum": chunk.checksum,
        "token_count": chunk.token_count,
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page

    for key, value in chunk.metadata.items():
        if is_scalar_metadata(value):
            metadata[f"meta_{key}"] = value

    return metadata


def parse_query_result(query_result: Mapping[str, Any]) -> tuple[VectorSearchResult, ...]:
    """Convert a Chroma query response into domain search results."""
    ids = first_result_list(query_result, "ids")
    documents = first_result_list(query_result, "documents")
    metadatas = first_result_list(query_result, "metadatas")
    distances = first_result_list(query_result, "distances")

    results: list[VectorSearchResult] = []
    for index, chunk_id in enumerate(ids):
        metadata = cast(dict[str, Any], metadatas[index] or {})
        distance = float(distances[index]) if index < len(distances) else 1.0
        document = str(documents[index]) if index < len(documents) else ""

        results.append(
            VectorSearchResult(
                chunk_id=str(metadata.get("chunk_id", chunk_id)),
                document_id=str(metadata["document_id"]),
                text=document,
                language=str(metadata.get("language", "unknown")),
                source=str(metadata.get("source", "")),
                chunk_index=int(metadata.get("chunk_index", index)),
                score=1.0 - distance,
                page=int(metadata["page"]) if "page" in metadata else None,
                token_count=int(metadata.get("token_count", 0)),
                metadata=extract_custom_metadata(metadata),
            )
        )

    return tuple(results)


def first_result_list(query_result: Mapping[str, Any], key: str) -> list[Any]:
    """Return the first query result list for a Chroma response key."""
    value = query_result.get(key) or []
    if not value:
        return []
    first = value[0]
    return list(first) if first is not None else []


def extract_custom_metadata(metadata: Mapping[str, Any]) -> dict[str, MetadataValue]:
    """Return metadata fields originally attached to a chunk."""
    custom_metadata: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if key.startswith("meta_") and is_scalar_metadata(value):
            custom_metadata[key.removeprefix("meta_")] = value
    return custom_metadata


def is_scalar_metadata(value: object) -> bool:
    """Return whether a value can be stored as Chroma scalar metadata."""
    return isinstance(value, str | int | float | bool)
