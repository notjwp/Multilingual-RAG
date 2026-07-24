"""Local FAISS vector store: a per-scope index file + a JSON metadata sidecar.

FAISS is an in-process library, not a server, so this adapter avoids embedded-Chroma's
multi-process pitfall (a long-lived client that can't see another process's writes) by holding no
shared live state: every operation loads the scope's current index file — cached by ``mtime`` and
reloaded when the worker rewrites it — and writes are atomic (temp + ``os.replace``) under a
cross-process ``FileLock``. So "worker writes → API reads" just works.

FAISS stores only vectors + int64 ids, so a sidecar JSON file holds each chunk's text and metadata.
Vectors are compared by inner product on the already-normalized bge-m3 embeddings, i.e. cosine
similarity — so ``score`` matches the Chroma adapter's semantics. Indexes are per **scope** — a
user, or a ``(user, chat)`` pair when a ``session_id`` is given (M18 per-chat documents) — which
makes both user and chat scoping leak-proof by construction and keeps each index small.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import status
from filelock import FileLock

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentChunk, VectorSearchResult
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.vectorstores.base import MetadataValue, VectorFilter

EMBEDDING_DIM = 1024  # bge-m3


class _UserIndex:
    """A user's loaded FAISS index + metadata, tagged with the file mtime it was read at."""

    def __init__(self, index: Any, meta: dict[str, dict[str, Any]], mtime: float | None) -> None:
        self.index = index
        self.meta = meta  # str(int64 id) -> chunk metadata dict
        self.mtime = mtime


class FaissVectorStore:
    """Persist and search document chunks in per-user FAISS index files."""

    def __init__(self, settings: Settings, *, dimension: int = EMBEDDING_DIM) -> None:
        self._dir = settings.faiss_persist_directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dim = dimension
        self._cache: dict[str, _UserIndex] = {}
        self._cache_lock = threading.Lock()  # guards self._cache within this process

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> None:
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
        scope = self._scope(user_id, session_id)
        with self._file_lock(scope):
            state = self._load(scope)
            ids = np.array([_chunk_int_id(c.chunk_id) for c in chunks], dtype=np.int64)
            state.index.remove_ids(ids)  # replace semantics; ignores ids not present
            vectors = _to_matrix([list(e) for e in embeddings])
            state.index.add_with_ids(vectors, ids)
            for chunk, int_id in zip(chunks, ids, strict=True):
                state.meta[str(int(int_id))] = _meta_for_chunk(chunk)
            self._persist(scope, state)

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
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
        state = self._load(self._scope(user_id, session_id))
        if state.index.ntotal == 0:
            return ()
        query = _to_matrix([list(query_embedding)])
        # Over-fetch when filtering so post-hoc filters can still fill top_k.
        fetch = min(top_k * 4 if filters else top_k, state.index.ntotal)
        scores, ids = state.index.search(query, fetch)

        results: list[VectorSearchResult] = []
        for score, int_id in zip(scores[0], ids[0], strict=True):
            if int(int_id) == -1:
                continue
            item = state.meta.get(str(int(int_id)))
            if item is None or (filters and not _matches(item, filters)):
                continue
            results.append(_to_result(item, float(score)))
            if len(results) >= top_k:
                break
        return tuple(results)

    def delete_document(
        self, document_id: str, *, user_id: str, session_id: str | None = None
    ) -> None:
        if not document_id.strip():
            raise AppError(
                "document_id is required.",
                code="missing_document_id",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        scope = self._scope(user_id, session_id)
        with self._file_lock(scope):
            state = self._load(scope)
            remove = [
                int(key)
                for key, item in state.meta.items()
                if item.get("document_id") == document_id
            ]
            if not remove:
                return
            state.index.remove_ids(np.array(remove, dtype=np.int64))
            state.meta = {
                key: item
                for key, item in state.meta.items()
                if item.get("document_id") != document_id
            }
            self._persist(scope, state)

    # --- scope / paths / locks ---
    def _scope(self, user_id: str, session_id: str | None) -> str:
        """One index file per user, or per ``(user, chat)`` when a session is given (M18)."""
        if session_id is None:
            return _safe(user_id)
        return f"{_safe(user_id)}__{_safe(session_id)}"

    def _index_path(self, scope: str) -> Path:
        return self._dir / f"{scope}.faiss"

    def _meta_path(self, scope: str) -> Path:
        return self._dir / f"{scope}.meta.json"

    def _file_lock(self, scope: str) -> FileLock:
        return FileLock(str(self._dir / f"{scope}.lock"))

    # --- load / persist ---
    def _load(self, scope: str) -> _UserIndex:
        index_path = self._index_path(scope)
        mtime = index_path.stat().st_mtime if index_path.exists() else None
        with self._cache_lock:
            cached = self._cache.get(scope)
            if cached is not None and cached.mtime == mtime:
                return cached

        if mtime is None:
            state = _UserIndex(_new_index(self._dim), {}, None)
        else:
            index = faiss.read_index(str(index_path))
            meta_path = self._meta_path(scope)
            meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
            state = _UserIndex(index, meta, mtime)

        with self._cache_lock:
            self._cache[scope] = state
        return state

    def _persist(self, scope: str, state: _UserIndex) -> None:
        index_path = self._index_path(scope)
        _atomic_write_index(index_path, state.index)
        _atomic_write_text(self._meta_path(scope), json.dumps(state.meta))
        state.mtime = index_path.stat().st_mtime
        with self._cache_lock:
            self._cache[scope] = state


def _new_index(dimension: int) -> Any:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dimension))


def _to_matrix(vectors: list[list[float]]) -> Any:
    matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)  # in-place → inner product == cosine similarity
    return matrix


def _chunk_int_id(chunk_id: str) -> int:
    """Stable signed 64-bit id for a chunk (unique within a per-user index)."""
    digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _safe(user_id: str) -> str:
    """Sanitize a user id into a safe file stem."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id) or "_"


def _meta_for_chunk(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "language": chunk.language,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "checksum": chunk.checksum,
        "token_count": chunk.token_count,
        "page": chunk.page,
        "meta": {
            key: value
            for key, value in chunk.metadata.items()
            if isinstance(value, str | int | float | bool)
        },
    }


def _matches(item: dict[str, Any], filters: VectorFilter) -> bool:
    custom = item.get("meta", {})
    for key, value in filters.items():
        actual = item.get(key, custom.get(key))
        if actual != value:
            return False
    return True


def _to_result(item: dict[str, Any], score: float) -> VectorSearchResult:
    custom: dict[str, MetadataValue] = {
        key: value
        for key, value in item.get("meta", {}).items()
        if isinstance(value, str | int | float | bool)
    }
    page = item.get("page")
    return VectorSearchResult(
        chunk_id=str(item["chunk_id"]),
        document_id=str(item["document_id"]),
        text=str(item.get("text", "")),
        language=str(item.get("language", "unknown")),
        source=str(item.get("source", "")),
        chunk_index=int(item.get("chunk_index", 0)),
        score=score,
        page=int(page) if page is not None else None,
        token_count=int(item.get("token_count", 0)),
        metadata=custom,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_index(path: Path, index: Any) -> None:
    tmp = f"{path}.tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, str(path))
