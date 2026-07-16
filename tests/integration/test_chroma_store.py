from pathlib import Path

import pytest

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentChunk
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore, metadata_for_chunk


def make_chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    text: str = "hello world",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        language="en",
        source="sample.txt",
        chunk_index=int(chunk_id.rsplit("-", maxsplit=1)[-1]),
        checksum=f"checksum-{chunk_id}",
        token_count=len(text.split()),
        metadata={"section_index": 1, "ignored": {"nested": True}},
    )


def test_chroma_store_upsert_search_and_delete(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        Settings(
            chroma_persist_directory=tmp_path / "chroma",
            chroma_collection_name="test_collection",
        )
    )
    chunks = (
        make_chunk("chunk-0", text="alpha document"),
        make_chunk("chunk-1", text="beta document"),
    )

    store.upsert_chunks(chunks, ([1.0, 0.0], [0.0, 1.0]), user_id="user-1")
    results = store.search(
        [1.0, 0.0], user_id="user-1", top_k=2, filters={"document_id": "doc-1"}
    )

    assert results[0].chunk_id == "chunk-0"
    assert results[0].metadata["section_index"] == 1
    assert len(results) == 2

    store.delete_document("doc-1", user_id="user-1")

    assert store.search([1.0, 0.0], user_id="user-1", top_k=2) == ()


def test_chroma_store_rejects_mismatched_upsert_sizes(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        Settings(
            chroma_persist_directory=tmp_path / "chroma",
            chroma_collection_name="test_collection",
        )
    )

    with pytest.raises(AppError, match="counts must match"):
        store.upsert_chunks((make_chunk("chunk-0"),), (), user_id="user-1")


def test_search_is_scoped_to_the_calling_user(tmp_path: Path) -> None:
    """The tenancy boundary: user A's search must never surface user B's chunks."""
    store = ChromaVectorStore(
        Settings(
            chroma_persist_directory=tmp_path / "chroma",
            chroma_collection_name="test_collection",
        )
    )
    store.upsert_chunks((make_chunk("chunk-0", text="alpha"),), ([1.0, 0.0],), user_id="user-A")
    store.upsert_chunks((make_chunk("chunk-0", text="beta"),), ([1.0, 0.0],), user_id="user-B")

    a_results = store.search([1.0, 0.0], user_id="user-A", top_k=10)
    b_results = store.search([1.0, 0.0], user_id="user-B", top_k=10)

    # Identical file (same chunk_id) uploaded by both users must NOT overwrite: each user
    # keeps their own copy, and neither sees the other's.
    assert [r.text for r in a_results] == ["alpha"]
    assert [r.text for r in b_results] == ["beta"]


def test_delete_is_scoped_to_the_calling_user(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        Settings(
            chroma_persist_directory=tmp_path / "chroma",
            chroma_collection_name="test_collection",
        )
    )
    store.upsert_chunks((make_chunk("chunk-0", text="alpha"),), ([1.0, 0.0],), user_id="user-A")
    store.upsert_chunks((make_chunk("chunk-0", text="beta"),), ([1.0, 0.0],), user_id="user-B")

    store.delete_document("doc-1", user_id="user-A")

    # A's document is gone; B's identical-id document survives.
    assert store.search([1.0, 0.0], user_id="user-A", top_k=10) == ()
    assert [r.text for r in store.search([1.0, 0.0], user_id="user-B", top_k=10)] == ["beta"]


def test_metadata_for_chunk_keeps_only_scalar_custom_metadata() -> None:
    metadata = metadata_for_chunk(make_chunk("chunk-0"), "user-1")

    assert metadata["user_id"] == "user-1"
    assert metadata["document_id"] == "doc-1"
    assert metadata["meta_section_index"] == 1
    assert "meta_ignored" not in metadata
