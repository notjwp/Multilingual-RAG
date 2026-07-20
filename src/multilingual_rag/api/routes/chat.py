"""Chat session routes: create/list/rename/delete sessions and send messages."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, cast

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.api.routes.query import get_query_service
from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.chat.repository import ChatSessionRepository, MessageRepository
from multilingual_rag.chat.service import ChatService, QueryAnswerer
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    AnswerCitation,
    ChatSessionRecord,
    MessageRecord,
    UserRecord,
)
from multilingual_rag.db.session import get_session

router = APIRouter(prefix="/v1/chats", tags=["chats"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)
SESSION_DEPENDENCY = Depends(get_session)


class CreateChatRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class SendMessageRequest(BaseModel):
    query: str = Field(min_length=1)


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    citations: tuple[AnswerCitation, ...] = ()


class ChatDetailResponse(BaseModel):
    session: ChatSessionResponse
    messages: tuple[MessageResponse, ...]


class ChatServiceProtocol(Protocol):
    """Protocol for chat route orchestration (see ``ChatService``)."""

    async def create_session(self, *, user_id: str, title: str | None = None) -> ChatSessionRecord:
        ...

    async def list_sessions(self, *, user_id: str) -> tuple[ChatSessionRecord, ...]:
        ...

    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        ...

    async def rename_session(
        self, *, user_id: str, session_id: str, title: str
    ) -> ChatSessionRecord:
        ...

    async def delete_session(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        ...

    async def send_message(self, *, user_id: str, session_id: str, query: str) -> MessageRecord:
        ...


@router.post("", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    body: CreateChatRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> ChatSessionResponse:
    """Create a new chat session."""
    record = await get_chat_service(request, session).create_session(
        user_id=current_user.user_id, title=body.title
    )
    await session.commit()
    return session_response(record)


@router.get("", response_model=tuple[ChatSessionResponse, ...])
async def list_chats(
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> tuple[ChatSessionResponse, ...]:
    """List the authenticated user's chat sessions."""
    records = await get_chat_service(request, session).list_sessions(user_id=current_user.user_id)
    return tuple(session_response(record) for record in records)


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(
    chat_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> ChatDetailResponse:
    """Return a chat session with its full message history."""
    record, messages = await get_chat_service(request, session).get_session(
        user_id=current_user.user_id, session_id=chat_id
    )
    return ChatDetailResponse(
        session=session_response(record),
        messages=tuple(message_response(message) for message in messages),
    )


@router.patch("/{chat_id}", response_model=ChatSessionResponse)
async def rename_chat(
    chat_id: str,
    body: RenameChatRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> ChatSessionResponse:
    """Rename a chat session."""
    record = await get_chat_service(request, session).rename_session(
        user_id=current_user.user_id, session_id=chat_id, title=body.title
    )
    await session.commit()
    return session_response(record)


@router.delete("/{chat_id}", response_model=ChatSessionResponse)
async def delete_chat(
    chat_id: str,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> ChatSessionResponse:
    """Delete a chat session and its messages."""
    record = await get_chat_service(request, session).delete_session(
        user_id=current_user.user_id, session_id=chat_id
    )
    await session.commit()
    return session_response(record)


@router.post("/{chat_id}/messages", response_model=MessageResponse)
async def send_message(
    chat_id: str,
    body: SendMessageRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> MessageResponse:
    """Send a user message and return the generated assistant reply with citations."""
    assistant = await get_chat_service(request, session).send_message(
        user_id=current_user.user_id, session_id=chat_id, query=body.query
    )
    await session.commit()
    return message_response(assistant)


def get_chat_service(request: Request, session: AsyncSession) -> ChatServiceProtocol:
    """Return an injected (test) or default chat service."""
    existing_service = getattr(request.app.state, "chat_service", None)
    if existing_service is not None:
        return cast(ChatServiceProtocol, existing_service)
    settings = cast(Settings, request.app.state.settings)
    return ChatService(
        session_repository=ChatSessionRepository(session),
        message_repository=MessageRepository(session),
        query_service=cast(QueryAnswerer, get_query_service(request)),
        history_max_messages=settings.chat_history_max_messages,
    )


def session_response(record: ChatSessionRecord) -> ChatSessionResponse:
    return ChatSessionResponse(
        session_id=record.session_id, title=record.title, created_at=record.created_at
    )


def message_response(record: MessageRecord) -> MessageResponse:
    return MessageResponse(
        message_id=record.message_id,
        role=record.role,
        content=record.content,
        created_at=record.created_at,
        citations=record.citations,
    )
