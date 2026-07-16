from pathlib import Path

from multilingual_rag.core.config import Settings
from multilingual_rag.documents.service import (
    DocumentIndexingService,
    sanitize_filename,
    save_upload_bytes,
)
from multilingual_rag.embeddings.base import EmbeddingVector
from multilingual_rag.ingestion.service import IngestionService
from multilingual_rag.storage.document_store import DocumentStore
from multilingual_rag.vectorstores.base import VectorFilter


class FakeEmbeddingProvider:
    def embed_documents(self, texts: tuple[str, ...]) -> list[EmbeddingVector]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> EmbeddingVector:
        return [float(len(text))]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upsert_count = 0
        self.upsert_user_ids: list[str] = []
        self.deleted: list[tuple[str, str]] = []

    def upsert_chunks(self, chunks: object, embeddings: object, *, user_id: str) -> None:
        del chunks, embeddings
        self.upsert_count += 1
        self.upsert_user_ids.append(user_id)

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        user_id: str,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> tuple:
        del query_embedding, user_id, top_k, filters
        return ()

    def delete_document(self, document_id: str, *, user_id: str) -> None:
        self.deleted.append((document_id, user_id))


def test_save_upload_bytes_sanitizes_and_persists_file(tmp_path: Path) -> None:
    saved_path = save_upload_bytes(
        tmp_path,
        filename="../unsafe name.txt",
        content=b"hello",
    )

    assert saved_path.read_bytes() == b"hello"
    assert saved_path.parent == tmp_path
    assert saved_path.name.endswith("unsafe_name.txt")


def test_sanitize_filename_falls_back_for_empty_names() -> None:
    assert sanitize_filename("...") == "document.txt"


def test_document_indexing_service_indexes_and_deletes_text_file(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("This document explains multilingual RAG. " * 3, encoding="utf-8")
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        ingestion_service=IngestionService(Settings(chunk_size_tokens=12, chunk_overlap_tokens=2)),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        document_store=DocumentStore(tmp_path / "documents.json"),
    )

    record = service.index_file(source_path, user_id="user-1")
    fetched = service.get_document(record.document.document_id)
    deleted = service.delete_document(record.document.document_id, user_id="user-1")

    assert fetched == record
    assert deleted == record
    assert vector_store.upsert_count == 1
    assert vector_store.upsert_user_ids == ["user-1"]
    assert vector_store.deleted == [(record.document.document_id, "user-1")]
