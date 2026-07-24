"""Chat orchestration: sessions + messages, wiring the RAG pipeline into stored turns."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from fastapi import status

from multilingual_rag.chat.repository import ChatSessionRepository, MessageRepository
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    ChatSessionRecord,
    ConversationTurn,
    GeneratedAnswer,
    MessageRecord,
)
from multilingual_rag.generation.streaming import Done, StreamEvent, Token

DEFAULT_TITLE = "New chat"
DEFAULT_HISTORY_MAX_MESSAGES = 10


class QueryAnswerer(Protocol):
    """The RAG orchestrator ChatService needs — satisfied by ``RagQueryService``."""

    def answer(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> GeneratedAnswer:
        """Retrieve context and generate a grounded answer, optionally with prior turns."""
        ...


class StreamingAnswerer(Protocol):
    """The streaming RAG orchestrator — satisfied by ``StreamingAnswerGenerator``."""

    def stream(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Retrieve context and stream a grounded answer as ``Token``s then a ``Done``."""
        ...


@dataclass(frozen=True)
class TokenChunk:
    """A streamed slice of the assistant's reply, forwarded to the client verbatim."""

    text: str


@dataclass(frozen=True)
class CompletedMessage:
    """The persisted assistant turn, emitted once streaming finishes."""

    message: MessageRecord


ChatStreamEvent = TokenChunk | CompletedMessage


class ChatService:
    """Create/list/rename/delete chat sessions and answer messages with the RAG pipeline."""

    def __init__(
        self,
        *,
        session_repository: ChatSessionRepository,
        message_repository: MessageRepository,
        query_service: QueryAnswerer,
        streaming_answerer: StreamingAnswerer | None = None,
        history_max_messages: int = DEFAULT_HISTORY_MAX_MESSAGES,
    ) -> None:
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.query_service = query_service
        self.streaming_answerer = streaming_answerer
        self.history_max_messages = history_max_messages

    async def _history(self, session_id: str) -> tuple[ConversationTurn, ...]:
        """The recent prior turns of a session, as conversation context for generation."""
        prior = await self.message_repository.list(session_id=session_id)
        recent = prior[-self.history_max_messages :] if self.history_max_messages else ()
        return tuple(ConversationTurn(role=m.role, content=m.content) for m in recent)

    async def create_session(self, *, user_id: str, title: str | None = None) -> ChatSessionRecord:
        return await self.session_repository.create(user_id=user_id, title=title or DEFAULT_TITLE)

    async def list_sessions(self, *, user_id: str) -> tuple[ChatSessionRecord, ...]:
        return await self.session_repository.list(user_id=user_id)

    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        session = await self.session_repository.get(user_id=user_id, session_id=session_id)
        messages = await self.message_repository.list(session_id=session_id)
        return session, messages

    async def rename_session(
        self, *, user_id: str, session_id: str, title: str
    ) -> ChatSessionRecord:
        return await self.session_repository.rename(
            user_id=user_id, session_id=session_id, title=title
        )

    async def delete_session(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        return await self.session_repository.delete(user_id=user_id, session_id=session_id)

    async def send_message(
        self, *, user_id: str, session_id: str, query: str
    ) -> MessageRecord:
        """Persist the user turn, run the RAG pipeline, persist + return the assistant turn."""
        session = await self.session_repository.get(user_id=user_id, session_id=session_id)
        history = await self._history(session_id)  # prior turns, before this one is stored
        await self.message_repository.add(session_id=session_id, role="user", content=query)
        # The RAG core is sync (local embeddings + a generation HTTP call) — offload it so it
        # doesn't stall the event loop (same as the /v1/query route).
        answer = await asyncio.to_thread(
            self.query_service.answer,
            query,
            user_id=user_id,
            session_id=session_id,
            history=history,
        )
        assistant = await self.message_repository.add(
            session_id=session_id,
            role="assistant",
            content=answer.answer,
            citations=answer.citations,
        )
        # Name a fresh session after its first message so the sidebar isn't a wall of "New chat".
        if session.title == DEFAULT_TITLE:
            await self.session_repository.rename(
                user_id=user_id, session_id=session_id, title=_derive_title(query)
            )
        return assistant

    async def stream_message(
        self, *, user_id: str, session_id: str, query: str
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream the answer token-by-token, persisting both turns once it completes.

        Forwards each ``Token`` as a ``TokenChunk``; on the final ``Done`` it persists the
        assistant turn (with citations), auto-titles a fresh session, and yields the persisted
        message as a ``CompletedMessage``.
        """
        if self.streaming_answerer is None:
            raise AppError(
                "Streaming is not configured for this chat service.",
                code="streaming_unavailable",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        session = await self.session_repository.get(user_id=user_id, session_id=session_id)
        history = await self._history(session_id)  # prior turns, before this one is stored
        await self.message_repository.add(session_id=session_id, role="user", content=query)

        generated: GeneratedAnswer | None = None
        async for event in self.streaming_answerer.stream(
            query, user_id=user_id, session_id=session_id, history=history
        ):
            if isinstance(event, Token):
                yield TokenChunk(event.text)
            elif isinstance(event, Done):
                generated = event.answer
        if generated is None:  # the generator raises on an empty answer, so this is defensive
            raise AppError(
                "The generation endpoint returned an empty answer.",
                code="empty_generation_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        assistant = await self.message_repository.add(
            session_id=session_id,
            role="assistant",
            content=generated.answer,
            citations=generated.citations,
        )
        if session.title == DEFAULT_TITLE:
            await self.session_repository.rename(
                user_id=user_id, session_id=session_id, title=_derive_title(query)
            )
        yield CompletedMessage(assistant)


def _derive_title(query: str) -> str:
    title = " ".join(query.split())[:60].strip()
    return title or DEFAULT_TITLE
