"""Database-backed chat session and message repositories.

Mirrors ``documents/repository.py``: async, user-scoped, raising ``AppError`` for a missing or
other-user session, and using Core bulk ``delete()`` (the DB ``ondelete="CASCADE"`` FKs remove the
messages and citations of a deleted chat).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from fastapi import status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import AnswerCitation, ChatSessionRecord, MessageRecord
from multilingual_rag.db.models import ChatSession, Message, MessageCitation


class ChatSessionRepository:
    """Persist user-scoped chat sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, user_id: str, title: str) -> ChatSessionRecord:
        row = ChatSession(user_id=user_id, title=title)
        self.session.add(row)
        await self.session.flush()
        return _session_record(row)

    async def list(self, *, user_id: str) -> tuple[ChatSessionRecord, ...]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
        )
        return tuple(_session_record(row) for row in result.scalars().all())

    async def get(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        row = await self.session.get(ChatSession, session_id)
        if row is None or row.user_id != user_id:
            raise AppError(
                f"Chat session not found: {session_id}",
                code="chat_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return _session_record(row)

    async def rename(self, *, user_id: str, session_id: str, title: str) -> ChatSessionRecord:
        await self.get(user_id=user_id, session_id=session_id)  # ownership check (404 otherwise)
        row = await self.session.get(ChatSession, session_id)
        assert row is not None  # just fetched above
        row.title = title
        await self.session.flush()
        return _session_record(row)

    async def delete(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        record = await self.get(user_id=user_id, session_id=session_id)
        await self.session.execute(
            delete(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )
        await self.session.flush()
        return record


class MessageRepository:
    """Persist chat messages and their citations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        citations: Sequence[AnswerCitation] = (),
    ) -> MessageRecord:
        message = Message(session_id=session_id, role=role, content=content)
        self.session.add(message)
        await self.session.flush()  # assigns message.id
        citation_rows = [
            MessageCitation(
                message_id=message.id,
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                source=citation.source,
                page=citation.page,
                text=citation.text,
            )
            for citation in citations
        ]
        self.session.add_all(citation_rows)
        await self.session.flush()
        return _message_record(message, citation_rows)

    async def list(self, *, session_id: str) -> tuple[MessageRecord, ...]:
        result = await self.session.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
        )
        messages = list(result.scalars().all())
        if not messages:
            return ()
        citations = await self.session.execute(
            select(MessageCitation).where(
                MessageCitation.message_id.in_([message.id for message in messages])
            )
        )
        by_message: dict[str, list[MessageCitation]] = defaultdict(list)
        for citation in citations.scalars().all():
            by_message[citation.message_id].append(citation)
        return tuple(
            _message_record(message, by_message.get(message.id, [])) for message in messages
        )


def _session_record(row: ChatSession) -> ChatSessionRecord:
    return ChatSessionRecord(
        session_id=row.id, user_id=row.user_id, title=row.title, created_at=row.created_at
    )


def _message_record(message: Message, citations: Sequence[MessageCitation]) -> MessageRecord:
    return MessageRecord(
        message_id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        citations=tuple(
            AnswerCitation(
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                source=citation.source,
                page=citation.page,
                text=citation.text or "",
            )
            for citation in citations
        ),
    )
