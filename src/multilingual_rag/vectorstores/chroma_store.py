"""ChromaDB vector store implementation."""

from __future__ import annotations

import contextlib
import os
import threading
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
    """Persist and search document chunks in ChromaDB.

    Embedded Chroma caches each process's index segments in memory, so a client opened *before*
    another process (the Celery worker) writes will silently miss — or fail to resolve ("Error
    finding id") — those rows. To stay correct across the separate API and worker processes, this
    adapter records the persist directory's newest mtime and, when it advances (another process
    wrote), drops Chroma's process-wide client cache and reopens, reloading segments from disk. All
    operations are serialized under a lock so a reopen never tears a client down mid-query.
    """

    def __init__(self, settings: Settings) -> None:
        self._dir = settings.chroma_persist_directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = settings.chroma_collection_name
        self._lock = threading.Lock()
        self._loaded_mtime = -1.0
        self._client: Any = None
        self._collection: Any = None
        self._open()

    def _open(self) -> None:
        """(Re)open the client from disk, dropping Chroma's process-wide cache first so another
        process's writes are read from disk rather than served from a stale in-memory segment."""
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self._loaded_mtime = self._dir_mtime()

    def _dir_mtime(self) -> float:
        """Newest mtime anywhere under the persist dir — the signal another process has written."""
        latest = 0.0
        for root, _dirs, files in os.walk(self._dir):
            for name in files:
                with contextlib.suppress(OSError):
                    latest = max(latest, os.stat(os.path.join(root, name)).st_mtime)
        return latest

    def _current(self) -> Any:
        """The live collection, reopened first if another process wrote. Caller holds the lock."""
        if self._dir_mtime() > self._loaded_mtime:
            self._open()
        return self._collection

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> None:
        """Insert or update one user's (and chat's, when given) chunk embeddings in ChromaDB."""
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
        # Storage ids are namespaced by user so two users uploading the byte-identical file
        # (same checksum -> same document_id -> same chunk_id) do not overwrite each other.
        with self._lock:
            self._current().upsert(
                ids=[storage_id(user_id, chunk.chunk_id, session_id) for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=chroma_embeddings,
                metadatas=[metadata_for_chunk(chunk, user_id, session_id) for chunk in chunks],
            )
            self._loaded_mtime = self._dir_mtime()  # our own write; don't reopen next op

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search one user's (and chat's, when given) chunks nearest to a query embedding."""
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
        with self._lock:
            query_result = self._current().query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=scoped_where(user_id, filters, session_id),
                include=["documents", "metadatas", "distances"],
            )
        return parse_query_result(cast(dict[str, Any], query_result))

    def delete_document(
        self, document_id: str, *, user_id: str, session_id: str | None = None
    ) -> None:
        """Delete all of one user's (and chat's, when given) chunks for a document."""
        if not document_id.strip():
            raise AppError(
                "document_id is required.",
                code="missing_document_id",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        conditions: list[dict[str, Any]] = [{"user_id": user_id}, {"document_id": document_id}]
        if session_id is not None:
            conditions.append({"session_id": session_id})
        with self._lock:
            self._current().delete(where={"$and": conditions})
            self._loaded_mtime = self._dir_mtime()


def storage_id(user_id: str, chunk_id: str, session_id: str | None = None) -> str:
    """Namespace a chunk's Chroma storage id by user (and chat), transparent to API results.

    Two chats holding the byte-identical file (same ``chunk_id``) must not overwrite each other,
    so the chat scope is folded into the storage id — mirroring the per-user namespacing.
    """
    if session_id is None:
        return f"{user_id}:{chunk_id}"
    return f"{user_id}:{session_id}:{chunk_id}"


def scoped_where(
    user_id: str, filters: VectorFilter | None, session_id: str | None = None
) -> dict[str, Any]:
    """Build a Chroma where clause pinned to one user (and chat), un-widenable by client filters.

    Client conditions are AND-ed *under* the user (and, when given, chat) scope, so a filter can
    only narrow the scoped chunks, never reach another user's or chat's.
    """
    conditions: list[dict[str, Any]] = [{"user_id": user_id}]
    if session_id is not None:
        conditions.append({"session_id": session_id})
    if filters:
        conditions.extend({key: value} for key, value in filters.items())
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def metadata_for_chunk(
    chunk: DocumentChunk, user_id: str, session_id: str | None = None
) -> dict[str, MetadataValue]:
    """Convert chunk metadata to Chroma-compatible scalar metadata, scoped to a user (and chat)."""
    metadata: dict[str, MetadataValue] = {
        "user_id": user_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "language": chunk.language,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "checksum": chunk.checksum,
        "token_count": chunk.token_count,
    }
    if session_id is not None:
        metadata["session_id"] = session_id
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
