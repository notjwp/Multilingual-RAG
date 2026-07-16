from pathlib import Path

import pytest

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentMetadata, DocumentRecord
from multilingual_rag.storage.document_store import DocumentStore


def make_record(document_id: str = "doc-1") -> DocumentRecord:
    return DocumentRecord(
        document=DocumentMetadata(
            document_id=document_id,
            source="sample.txt",
            content_type="text/plain",
            checksum="abc123",
            language="en",
        ),
        chunk_count=2,
    )


def test_document_store_saves_gets_lists_and_deletes_records(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "documents.json")
    record = make_record()

    store.save(record)

    assert store.get("doc-1") == record
    assert store.list() == (record,)
    assert store.delete("doc-1") == record
    assert store.list() == ()


def test_document_store_raises_for_missing_document(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "documents.json")

    with pytest.raises(AppError, match="Document not found"):
        store.get("missing")

