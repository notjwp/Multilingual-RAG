"""Chat orchestration: sessions + messages, wiring the RAG pipeline into stored turns."""

from __future__ import annotations

import asyncio
from typing import Protocol

from multilingual_rag.chat.repository import ChatSessionRepository, MessageRepository
from multilingual_rag.core.models import ChatSessionRecord, GeneratedAnswer, MessageRecord

DEFAULT_TITLE = "New chat"


class QueryAnswerer(Protocol):
    """The RAG orchestrator ChatService needs — satisfied by ``RagQueryService``."""

    def answer(self, query: str, *, user_id: str) -> GeneratedAnswer:
        """Retrieve context and generate a grounded answer."""
        ...


class ChatService:
    """Create/list/rename/delete chat sessions and answer messages with the RAG pipeline."""

    def __init__(
        self,
        *,
        session_repository: ChatSessionRepository,
        message_repository: MessageRepository,
        query_service: QueryAnswerer,
    ) -> None:
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.query_service = query_service

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
        await self.message_repository.add(session_id=session_id, role="user", content=query)
        # The RAG core is sync (local embeddings + a generation HTTP call) — offload it so it
        # doesn't stall the event loop (same as the /v1/query route).
        answer = await asyncio.to_thread(self.query_service.answer, query, user_id=user_id)
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


def _derive_title(query: str) -> str:
    title = " ".join(query.split())[:60].strip()
    return title or DEFAULT_TITLE
