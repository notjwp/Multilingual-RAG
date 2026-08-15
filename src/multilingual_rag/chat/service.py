"""Chat orchestration: sessions + messages, wiring the RAG pipeline into stored turns."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from fastapi import status

from multilingual_rag.agent.events import AgentEvent, Done, Step, Token
from multilingual_rag.agent.state import AgentResult
from multilingual_rag.chat.repository import ChatSessionRepository, MessageRepository
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    ChatSessionRecord,
    ConversationTurn,
    GeneratedAnswer,
    MessageRecord,
)

DEFAULT_TITLE = "New chat"
DEFAULT_HISTORY_MAX_MESSAGES = 10


class RagAnswerer(Protocol):
    """The RAG orchestrator ChatService needs — satisfied by ``RagGraph``.

    One protocol for both paths, because there is now one orchestrator. Previously this was two
    (``QueryAnswerer`` + ``StreamingAnswerer``) pointing at two separate implementations.
    """

    async def answer(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AgentResult:
        """Retrieve context and generate a grounded answer, optionally with prior turns."""
        ...

    def stream(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[AgentEvent]:
        """Stream agent steps and answer tokens, then the assembled ``Done``."""
        ...


@dataclass(frozen=True)
class TokenChunk:
    """A streamed slice of the assistant's reply, forwarded to the client verbatim."""

    text: str


@dataclass(frozen=True)
class StepChunk:
    """One agent step, forwarded live to the client. Never persisted."""

    step: Step


@dataclass(frozen=True)
class CompletedMessage:
    """The persisted assistant turn, emitted once streaming finishes."""

    message: MessageRecord


ChatStreamEvent = TokenChunk | StepChunk | CompletedMessage


class ChatService:
    """Create/list/rename/delete chat sessions and answer messages with the RAG pipeline."""

    def __init__(
        self,
        *,
        session_repository: ChatSessionRepository,
        message_repository: MessageRepository,
        answerer: RagAnswerer,
        history_max_messages: int = DEFAULT_HISTORY_MAX_MESSAGES,
    ) -> None:
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.answerer = answerer
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
        # No to_thread here: the graph is async and offloads its own blocking calls per node.
        result = await self.answerer.answer(
            query, user_id=user_id, session_id=session_id, history=history
        )
        assistant = await self.message_repository.add(
            session_id=session_id,
            role="assistant",
            content=result.answer.answer,
            citations=result.answer.citations,
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

        Forwards each ``Step`` as a ``StepChunk`` and each ``Token`` as a ``TokenChunk``; on the
        final ``Done`` it persists the assistant turn (with citations), auto-titles a fresh
        session, and yields the persisted message as a ``CompletedMessage``.

        Steps are **ephemeral** — forwarded live and never written to the database, so reloading
        a chat shows the answer and its citations, nothing else.
        """
        session = await self.session_repository.get(user_id=user_id, session_id=session_id)
        history = await self._history(session_id)  # prior turns, before this one is stored
        await self.message_repository.add(session_id=session_id, role="user", content=query)

        generated: GeneratedAnswer | None = None
        async for event in self.answerer.stream(
            query, user_id=user_id, session_id=session_id, history=history
        ):
            if isinstance(event, Token):
                yield TokenChunk(event.text)
            elif isinstance(event, Step):
                yield StepChunk(event)
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
