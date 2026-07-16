"""Database-backed document and ingestion job repositories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from fastapi import status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import DocumentMetadata, DocumentRecord, IngestionJobRecord
from multilingual_rag.db.models import (
    Document,
    DocumentChunk,
    DocumentFile,
    IngestionJob,
    IngestionStatus,
)
from multilingual_rag.ingestion.service_utils import checksum_text


class DocumentRepository:
    """Persist user-scoped document metadata in PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(
        self,
        record: DocumentRecord,
        *,
        user_id: str,
        file_path: Path,
        filename: str,
        file_size_bytes: int,
        chunk_metadata: list[dict[str, object]],
    ) -> None:
        """Insert or replace document, file, and chunk metadata."""
        document_id = record.document.document_id
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        await self.session.execute(
            delete(DocumentFile).where(DocumentFile.document_id == document_id)
        )
        await self.session.execute(delete(Document).where(Document.id == document_id))

        document = Document(
            id=record.document.document_id,
            user_id=user_id,
            source=record.document.source,
            content_type=record.document.content_type,
            checksum=record.document.checksum,
            language=record.document.language,
            chunk_count=record.chunk_count,
            ingestion_status=IngestionStatus.SUCCEEDED,
            extra=record.document.extra,
        )
        self.session.add(document)
        self.session.add(
            DocumentFile(
                document_id=record.document.document_id,
                path=str(file_path),
                filename=filename,
                content_type=record.document.content_type,
                size_bytes=file_size_bytes,
                checksum=checksum_text(str(file_path)),
            )
        )
        self.session.add_all(
            DocumentChunk(
                document_id=document_id,
                chunk_id=str(metadata["chunk_id"]),
                chunk_index=metadata_int(metadata, "chunk_index"),
                language=str(metadata["language"]),
                source=str(metadata["source"]),
                page=metadata_int(metadata, "page") if metadata.get("page") is not None else None,
                token_count=metadata_int(metadata, "token_count"),
                checksum=str(metadata["checksum"]),
                chunk_metadata=dict(cast(Mapping[str, object], metadata.get("metadata", {}))),
            )
            for metadata in chunk_metadata
        )
        await self.session.flush()

    async def get(self, *, user_id: str, document_id: str) -> DocumentRecord:
        """Return one user-scoped document."""
        document = await self.session.get(Document, document_id)
        if document is None or document.user_id != user_id:
            raise AppError(
                f"Document not found: {document_id}",
                code="document_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return document_record_from_orm(document)

    async def list(self, *, user_id: str) -> tuple[DocumentRecord, ...]:
        """Return all documents for a user."""
        result = await self.session.execute(
            select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
        )
        return tuple(document_record_from_orm(document) for document in result.scalars().all())

    async def delete(self, *, user_id: str, document_id: str) -> DocumentRecord:
        """Delete one user-scoped document and return its previous record."""
        record = await self.get(user_id=user_id, document_id=document_id)
        await self.session.execute(
            delete(Document).where(Document.id == document_id, Document.user_id == user_id)
        )
        await self.session.flush()
        return record


class IngestionJobRepository:
    """Persist ingestion job state."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, user_id: str, file_path: Path) -> IngestionJobRecord:
        """Create a queued ingestion job."""
        job = IngestionJob(user_id=user_id, file_path=str(file_path), status=IngestionStatus.QUEUED)
        self.session.add(job)
        await self.session.flush()
        return ingestion_job_record_from_orm(job)

    async def get(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        """Return a user-scoped ingestion job."""
        job = await self.session.get(IngestionJob, job_id)
        if job is None or job.user_id != user_id:
            raise AppError(
                f"Ingestion job not found: {job_id}",
                code="ingestion_job_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return ingestion_job_record_from_orm(job)

    async def mark_running(self, job_id: str) -> None:
        """Mark a job as running."""
        job = await self.session.get(IngestionJob, job_id)
        if job is not None:
            job.status = IngestionStatus.RUNNING
            await self.session.flush()

    async def mark_succeeded(self, *, job_id: str, document_id: str) -> None:
        """Mark a job as succeeded."""
        job = await self.session.get(IngestionJob, job_id)
        if job is not None:
            job.status = IngestionStatus.SUCCEEDED
            job.document_id = document_id
            await self.session.flush()

    async def mark_failed(self, *, job_id: str, error_message: str) -> None:
        """Mark a job as failed."""
        job = await self.session.get(IngestionJob, job_id)
        if job is not None:
            job.status = IngestionStatus.FAILED
            job.error_message = error_message
            await self.session.flush()


def document_record_from_orm(document: Document) -> DocumentRecord:
    """Convert an ORM document row to a domain record."""
    return DocumentRecord(
        document=DocumentMetadata(
            document_id=document.id,
            source=document.source,
            content_type=document.content_type,
            checksum=document.checksum,
            language=document.language,
            created_at=document.created_at,
            extra=document.extra,
        ),
        chunk_count=document.chunk_count,
    )


def ingestion_job_record_from_orm(job: IngestionJob) -> IngestionJobRecord:
    """Convert an ORM ingestion job row to a domain record."""
    return IngestionJobRecord(
        job_id=job.id,
        user_id=job.user_id,
        file_path=job.file_path,
        status=job.status,
        document_id=job.document_id,
        error_message=job.error_message,
    )


def metadata_int(metadata: dict[str, object], key: str) -> int:
    """Read an integer value from chunk metadata."""
    value = metadata[key]
    if not isinstance(value, str | int | float):
        raise TypeError(f"metadata field {key} must be numeric")
    return int(value)
