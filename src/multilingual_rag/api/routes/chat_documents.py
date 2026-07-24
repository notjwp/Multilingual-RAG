"""Chat-scoped document routes (M18).

Documents are uploaded *into a chat* and are retrievable only by that chat. Each route first
verifies the caller owns the chat (``get_session`` raises ``chat_not_found`` otherwise), then
reuses the shared document-service plumbing from ``documents.py`` with ``session_id=chat_id``.
"""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.api.routes.chat import get_chat_service
from multilingual_rag.api.routes.documents import (
    DocumentResponse,
    IngestionJobResponse,
    document_response,
    enqueue_ingestion_job,
    get_document_service,
    ingestion_job_response,
)
from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import UserRecord
from multilingual_rag.db.session import get_session
from multilingual_rag.documents.service import save_upload_bytes

router = APIRouter(prefix="/v1/chats", tags=["chat-documents"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)
SESSION_DEPENDENCY = Depends(get_session)


@router.post("/{chat_id}/documents", response_model=IngestionJobResponse)
async def upload_chat_document(
    chat_id: str,
    file: UploadFile,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> IngestionJobResponse:
    """Upload a document into one chat and enqueue ingestion scoped to that chat."""
    settings = cast(Settings, request.app.state.settings)
    # Ownership: 404 if the chat isn't the caller's — never index into someone else's chat.
    await get_chat_service(request, session).get_session(
        user_id=current_user.user_id, session_id=chat_id
    )
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
        session_id=chat_id,
        path=saved_path,
    )
    enqueue_ingestion_job(request, job.job_id)
    await session.commit()
    return ingestion_job_response(job)


@router.get("/{chat_id}/documents", response_model=tuple[DocumentResponse, ...])
async def list_chat_documents(
    chat_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> tuple[DocumentResponse, ...]:
    """Return the documents uploaded into one chat."""
    await get_chat_service(request, session).get_session(
        user_id=current_user.user_id, session_id=chat_id
    )
    records = await get_document_service(request, session).list_documents(
        user_id=current_user.user_id,
        session_id=chat_id,
    )
    return tuple(document_response(record) for record in records)


@router.delete("/{chat_id}/documents/{document_id}", response_model=DocumentResponse)
async def delete_chat_document(
    chat_id: str,
    document_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> DocumentResponse:
    """Delete one document from a chat (metadata + the chat-scoped vectors)."""
    await get_chat_service(request, session).get_session(
        user_id=current_user.user_id, session_id=chat_id
    )
    record = await get_document_service(request, session).delete_document(
        user_id=current_user.user_id,
        document_id=document_id,
        session_id=chat_id,
    )
    await session.commit()
    return document_response(record)
