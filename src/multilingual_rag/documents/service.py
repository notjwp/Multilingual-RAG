"""Document indexing orchestration."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import status

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentRecord, IngestionJobRecord
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository
from multilingual_rag.vectorstores.base import VectorStore

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class DatabaseDocumentIndexingService:
    """Database-backed user-scoped document service."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        job_repository: IngestionJobRepository,
        vector_store: VectorStore,
    ) -> None:
        self.document_repository = document_repository
        self.job_repository = job_repository
        self.vector_store = vector_store

    async def create_ingestion_job(
        self, *, user_id: str, session_id: str | None = None, path: Path
    ) -> IngestionJobRecord:
        """Create an ingestion job for a saved uploaded file, scoped to a chat."""
        return await self.job_repository.create(
            user_id=user_id, session_id=session_id, file_path=path
        )

    async def get_ingestion_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        """Return a user-scoped ingestion job."""
        return await self.job_repository.get(user_id=user_id, job_id=job_id)

    async def list_documents(
        self, *, user_id: str, session_id: str | None = None
    ) -> tuple[DocumentRecord, ...]:
        """Return a user's documents, narrowed to one chat when ``session_id`` is given."""
        return await self.document_repository.list(user_id=user_id, session_id=session_id)

    async def delete_document(
        self, *, user_id: str, document_id: str, session_id: str | None = None
    ) -> DocumentRecord:
        """Delete one indexed document from metadata and the chat-scoped vector store."""
        record = await self.document_repository.delete(user_id=user_id, document_id=document_id)
        self.vector_store.delete_document(document_id, user_id=user_id, session_id=session_id)
        return record


def save_upload_bytes(raw_directory: Path, *, filename: str | None, content: bytes) -> Path:
    """Persist uploaded bytes under a sanitized unique file name."""
    if not content:
        raise AppError(
            "Uploaded document is empty.",
            code="empty_upload",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    raw_directory.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(filename or "document.txt")
    destination = raw_directory / f"{uuid4().hex}_{safe_name}"
    destination.write_bytes(content)
    return destination


def sanitize_filename(filename: str) -> str:
    """Return a path-safe file name while preserving the extension."""
    candidate = Path(filename).name.strip() or "document.txt"
    sanitized = SAFE_FILENAME_PATTERN.sub("_", candidate)
    return sanitized.strip("._") or "document.txt"
