"""Document upload and metadata routes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from fastapi import APIRouter, Depends, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    DocumentMetadata,
    DocumentRecord,
    IngestionJobRecord,
    UserRecord,
)
from multilingual_rag.db.session import get_session
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository
from multilingual_rag.documents.service import DatabaseDocumentIndexingService, save_upload_bytes
from multilingual_rag.vectorstores.factory import build_vector_store
from multilingual_rag.workers.celery_app import ingest_document

router = APIRouter(prefix="/v1/documents", tags=["documents"])
jobs_router = APIRouter(prefix="/v1/ingestion-jobs", tags=["ingestion-jobs"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)
SESSION_DEPENDENCY = Depends(get_session)


class DocumentResponse(BaseModel):
    """Public document metadata response."""

    document: DocumentMetadata
    chunk_count: int


class IngestionJobResponse(BaseModel):
    """Public ingestion job response."""

    job_id: str
    status: str
    document_id: str | None = None
    error_message: str | None = None


class DocumentService(Protocol):
    """Protocol for document route orchestration."""

    async def create_ingestion_job(self, *, user_id: str, path: Path) -> IngestionJobRecord:
        """Create an ingestion job."""
        ...

    async def get_ingestion_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        """Return one ingestion job."""
        ...

    async def list_documents(self, *, user_id: str) -> tuple[DocumentRecord, ...]:
        """Return indexed documents."""
        ...

    async def get_document(self, *, user_id: str, document_id: str) -> DocumentRecord:
        """Return one indexed document."""
        ...

    async def delete_document(self, *, user_id: str, document_id: str) -> DocumentRecord:
        """Delete one indexed document."""
        ...


@router.post("/upload", response_model=IngestionJobResponse)
async def upload_document(
    file: UploadFile,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> IngestionJobResponse:
    """Upload a document and enqueue asynchronous ingestion."""
    settings = cast(Settings, request.app.state.settings)
    # Read at most one byte past the cap: nothing larger than the limit ever enters memory.
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise AppError(
            "Uploaded document exceeds the size limit.",
            code="upload_too_large",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    saved_path = save_upload_bytes(
        settings.raw_document_directory,
        filename=file.filename,
        content=content,
    )
    job = await get_document_service(request, session).create_ingestion_job(
        user_id=current_user.user_id,
        path=saved_path,
    )
    enqueue_ingestion_job(request, job.job_id)
    await session.commit()
    return ingestion_job_response(job)


@router.get("", response_model=tuple[DocumentResponse, ...])
async def list_documents(
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> tuple[DocumentResponse, ...]:
    """Return all documents for the authenticated user."""
    records = await get_document_service(request, session).list_documents(
        user_id=current_user.user_id,
    )
    return tuple(document_response(record) for record in records)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> DocumentResponse:
    """Return stored metadata for an indexed document."""
    record = await get_document_service(request, session).get_document(
        user_id=current_user.user_id,
        document_id=document_id,
    )
    return document_response(record)


@router.delete("/{document_id}", response_model=DocumentResponse)
async def delete_document(
    document_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> DocumentResponse:
    """Delete an indexed document."""
    record = await get_document_service(request, session).delete_document(
        user_id=current_user.user_id,
        document_id=document_id,
    )
    await session.commit()
    return document_response(record)


@jobs_router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    job_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> IngestionJobResponse:
    """Return an ingestion job for the authenticated user."""
    job = await get_document_service(request, session).get_ingestion_job(
        user_id=current_user.user_id,
        job_id=job_id,
    )
    return ingestion_job_response(job)


def get_document_service(request: Request, session: AsyncSession) -> DocumentService:
    """Return an injected or default document indexing service."""
    existing_service = getattr(request.app.state, "document_service", None)
    if existing_service is not None:
        return cast(DocumentService, existing_service)

    settings = cast(Settings, request.app.state.settings)
    return DatabaseDocumentIndexingService(
        document_repository=DocumentRepository(session),
        job_repository=IngestionJobRepository(session),
        vector_store=build_vector_store(settings),
    )


def enqueue_ingestion_job(request: Request, job_id: str) -> None:
    """Enqueue an ingestion job, allowing tests to inject a fake enqueue function."""
    injected_enqueue = getattr(request.app.state, "enqueue_ingestion", None)
    if injected_enqueue is not None:
        cast(Callable[[str], None], injected_enqueue)(job_id)
        return
    ingest_document.delay(job_id)


def document_response(record: DocumentRecord) -> DocumentResponse:
    """Map a stored document record to an API response."""
    return DocumentResponse(document=record.document, chunk_count=record.chunk_count)


def ingestion_job_response(record: IngestionJobRecord) -> IngestionJobResponse:
    """Map an ingestion job record to an API response."""
    return IngestionJobResponse(
        job_id=record.job_id,
        status=record.status,
        document_id=record.document_id,
        error_message=record.error_message,
    )
