"""Streaming chat route: Server-Sent Events for token-by-token answers.

Companion to ``chat.py``'s blocking ``POST /v1/chats/{id}/messages``. The ownership check runs
*before* the response starts so a missing/other-user chat is a clean 404 JSON — once the
``StreamingResponse`` begins, headers are sent and the only way to report a failure is an SSE
``error`` event.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Protocol, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.api.dependencies import get_rag_graph
from multilingual_rag.api.routes.chat import SendMessageRequest
from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.chat.repository import ChatSessionRepository, MessageRepository
from multilingual_rag.chat.service import (
    ChatService,
    ChatStreamEvent,
    CompletedMessage,
    StepChunk,
    TokenChunk,
)
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import ChatSessionRecord, MessageRecord, UserRecord
from multilingual_rag.db.session import get_session

router = APIRouter(prefix="/v1/chats", tags=["chats"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)
SESSION_DEPENDENCY = Depends(get_session)


class StreamingChatServiceProtocol(Protocol):
    """The chat orchestration the streaming route needs (see ``ChatService``)."""

    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        ...

    def stream_message(
        self, *, user_id: str, session_id: str, query: str
    ) -> AsyncIterator[ChatStreamEvent]:
        ...


@router.post("/{chat_id}/messages/stream")
async def stream_message(
    chat_id: str,
    body: SendMessageRequest,
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> StreamingResponse:
    """Stream a grounded answer as SSE, persisting both turns when the stream completes."""
    service = get_streaming_chat_service(request, session)
    # Verify ownership before any bytes are sent, so a missing chat is a 404 (not a 200 stream).
    await service.get_session(user_id=current_user.user_id, session_id=chat_id)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.stream_message(
                user_id=current_user.user_id, session_id=chat_id, query=body.query
            ):
                if isinstance(event, TokenChunk):
                    yield _sse(data={"token": event.text})
                elif isinstance(event, StepChunk):
                    # Ephemeral: shown live, never persisted. Clients that don't know this event
                    # ignore it — the frontend's frame dispatcher drops unknown event names.
                    yield _sse(event="step", data=asdict(event.step))
                elif isinstance(event, CompletedMessage):
                    await session.commit()
                    yield _sse(
                        event="done",
                        data={
                            "message_id": event.message.message_id,
                            "citations": [
                                citation.model_dump() for citation in event.message.citations
                            ],
                        },
                    )
        except AppError as exc:
            # Headers are already sent, so a mid-stream failure surfaces as an SSE error event.
            yield _sse(event="error", data={"error": exc.code, "message": exc.message})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def get_streaming_chat_service(
    request: Request, session: AsyncSession
) -> StreamingChatServiceProtocol:
    """Return an injected (test) chat service, or one wired with a streaming answerer."""
    existing_service = getattr(request.app.state, "chat_service", None)
    if existing_service is not None:
        return cast(StreamingChatServiceProtocol, existing_service)
    settings = cast(Settings, request.app.state.settings)
    return ChatService(
        session_repository=ChatSessionRepository(session),
        message_repository=MessageRepository(session),
        answerer=get_rag_graph(request),
        history_max_messages=settings.chat_history_max_messages,
    )


def _sse(*, data: dict[str, object], event: str | None = None) -> str:
    """Format one Server-Sent Event frame."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"
