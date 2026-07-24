"""FaissVectorStore: upsert/search/delete, user scoping, and cross-instance (worker→API) refresh."""

from __future__ import annotations

from pathlib import Path

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentChunk
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.vectorstores.faiss_store import FaissVectorStore


def _chunk(
    chunk_id: str, *, document_id: str = "doc-1", text: str = "x", index: int = 0
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        language="en",
        source="s.txt",
        chunk_index=index,
        checksum="ck",
        page=None,
        token_count=3,
        metadata={},
    )


def _store(tmp_path: Path) -> FaissVectorStore:
    settings = Settings(environment="test", vector_store="faiss", faiss_persist_directory=tmp_path)
    return FaissVectorStore(settings, dimension=4)


def test_upsert_and_search_returns_nearest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = [_chunk("doc-1:0", text="alpha", index=0), _chunk("doc-1:1", text="beta", index=1)]
    embeddings: list[EmbeddingVector] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.upsert_chunks(chunks, embeddings, user_id="u1")

    results = store.search([1.0, 0.0, 0.0, 0.0], user_id="u1", top_k=2)

    assert [r.chunk_id for r in results] == ["doc-1:0", "doc-1:1"]  # nearest first
    assert results[0].text == "alpha"
    assert results[0].score > results[1].score


def test_search_is_user_scoped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_chunks([_chunk("d:0", text="u1 doc")], [[1.0, 0.0, 0.0, 0.0]], user_id="u1")
    store.upsert_chunks([_chunk("d:0", text="u2 doc")], [[1.0, 0.0, 0.0, 0.0]], user_id="u2")

    results = store.search([1.0, 0.0, 0.0, 0.0], user_id="u1", top_k=5)
    assert [r.text for r in results] == ["u1 doc"]  # never another user's chunk


def test_delete_document_removes_only_that_document(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_chunks(
        [_chunk("a:0", document_id="a"), _chunk("b:0", document_id="b")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        user_id="u1",
    )

    store.delete_document("a", user_id="u1")

    results = store.search([1.0, 0.0, 0.0, 0.0], user_id="u1", top_k=5)
    assert all(r.document_id != "a" for r in results)
    assert any(r.document_id == "b" for r in results)


def test_reader_instance_sees_writer_instance_writes(tmp_path: Path) -> None:
    # Two instances over the same dir = the worker (writer) and the API (reader) processes.
    writer = _store(tmp_path)
    reader = _store(tmp_path)

    assert reader.search([1.0, 0.0, 0.0, 0.0], user_id="u1", top_k=5) == ()  # empty; caches mtime

    writer.upsert_chunks([_chunk("x:0", text="written")], [[1.0, 0.0, 0.0, 0.0]], user_id="u1")

    # The reader must pick up the writer's file (mtime changed → reload) — the Chroma pain point.
    results = reader.search([1.0, 0.0, 0.0, 0.0], user_id="u1", top_k=5)
    assert [r.text for r in results] == ["written"]
