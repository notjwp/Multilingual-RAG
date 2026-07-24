"""Ingestion-job status route + shared document-service plumbing.

Document upload/list/delete are **chat-scoped** in M18 and live in ``chat_documents.py``; this
module keeps the ingestion-job polling route and the pieces both routers share (the
``DocumentService`` protocol, the ``app.state`` service seam, the Celery enqueue helper, and the
response models/mappers).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    DocumentMetadata,
    DocumentRecord,
    IngestionJobRecord,
    UserRecord,
)
from multilingual_rag.db.session import get_session
from multilingual_rag.documents.repository import DocumentRepository, IngestionJobRepository
from multilingual_rag.documents.service import DatabaseDocumentIndexingService
from multilingual_rag.vectorstores.factory import build_vector_store
from multilingual_rag.workers.celery_app import ingest_document

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
    """Protocol for document route orchestration (chat-scoped in M18)."""

    async def create_ingestion_job(
        self, *, user_id: str, session_id: str | None = None, path: Path
    ) -> IngestionJobRecord:
        """Create an ingestion job, scoped to a chat."""
        ...

    async def get_ingestion_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        """Return one ingestion job."""
        ...

    async def list_documents(
        self, *, user_id: str, session_id: str | None = None
    ) -> tuple[DocumentRecord, ...]:
        """Return indexed documents, narrowed to a chat when ``session_id`` is given."""
        ...

    async def delete_document(
        self, *, user_id: str, document_id: str, session_id: str | None = None
    ) -> DocumentRecord:
        """Delete one indexed document."""
        ...


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
