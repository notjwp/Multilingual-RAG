"""ChatService unit tests with fake repositories — no database, no graph.

Covers the orchestration ChatService owns: history windowing, turn ordering, auto-titling, and
forwarding agent steps. The Postgres-backed version of some of this lives in
tests/integration/test_db_layer.py, which is skipped whenever Postgres is unreachable — these run
always.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

from multilingual_rag.agent.events import AgentEvent, Done, Step, Token
from multilingual_rag.agent.state import AgentResult
from multilingual_rag.chat.service import (
    DEFAULT_TITLE,
    ChatService,
    ChatStreamEvent,
    CompletedMessage,
    StepChunk,
    TokenChunk,
)
from multilingual_rag.core.models import (
    AnswerCitation,
    ChatSessionRecord,
    ConversationTurn,
    GeneratedAnswer,
    MessageRecord,
    RetrievalContext,
)


class FakeSessionRepository:
    def __init__(self, title: str = DEFAULT_TITLE) -> None:
        self.record = ChatSessionRecord(
            session_id="chat-1", user_id="user-1", title=title, created_at=datetime.now(UTC)
        )
        self.renames: list[str] = []

    async def get(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        return self.record

    async def rename(self, *, user_id: str, session_id: str, title: str) -> ChatSessionRecord:
        self.renames.append(title)
        return self.record


class FakeMessageRepository:
    def __init__(self) -> None:
        self.messages: list[MessageRecord] = []

    async def add(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        citations: Sequence[AnswerCitation] = (),
    ) -> MessageRecord:
        record = MessageRecord(
            message_id=f"m{len(self.messages)}",
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.now(UTC),
            citations=tuple(citations),
        )
        self.messages.append(record)
        return record

    async def list(self, *, session_id: str) -> tuple[MessageRecord, ...]:
        return tuple(self.messages)


class FakeAnswerer:
    """Stands in for RagGraph. Records the history it was handed."""

    def __init__(self, *, events: Sequence[AgentEvent] | None = None) -> None:
        self._events = events
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def _answer(self, query: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=f"reply: {query}",
            language="en",
            citations=(
                AnswerCitation(
                    chunk_id="c1", document_id="d1", source="s.txt", page=1, text="snippet"
                ),
            ),
        )

    async def answer(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AgentResult:
        self.histories.append(tuple(history))
        return AgentResult(
            answer=self._answer(query),
            context=RetrievalContext(query=query, query_language="en", results=()),
        )

    async def stream(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[AgentEvent]:
        self.histories.append(tuple(history))
        events = self._events
        if events is None:
            events = (Token("reply: "), Token(query), Done(self._answer(query)))
        for event in events:
            yield event


def _service(
    *, answerer: FakeAnswerer | None = None, title: str = DEFAULT_TITLE, history_max: int = 10
) -> tuple[ChatService, FakeSessionRepository, FakeMessageRepository, FakeAnswerer]:
    sessions = FakeSessionRepository(title)
    messages = FakeMessageRepository()
    graph = answerer or FakeAnswerer()
    service = ChatService(
        session_repository=sessions,  # type: ignore[arg-type]
        message_repository=messages,  # type: ignore[arg-type]
        answerer=graph,
        history_max_messages=history_max,
    )
    return service, sessions, messages, graph


async def _collect(generator: AsyncIterator[ChatStreamEvent]) -> list[ChatStreamEvent]:
    return [event async for event in generator]


# --- blocking path -----------------------------------------------------------------------------


def test_send_message_persists_both_turns_in_order() -> None:
    service, _, messages, _ = _service()

    assistant = asyncio.run(
        service.send_message(user_id="user-1", session_id="chat-1", query="what is bharat")
    )

    assert [(m.role, m.content) for m in messages.messages] == [
        ("user", "what is bharat"),
        ("assistant", "reply: what is bharat"),
    ]
    assert assistant.citations[0].chunk_id == "c1"


def test_send_message_auto_titles_a_fresh_chat() -> None:
    service, sessions, _, _ = _service(title=DEFAULT_TITLE)

    asyncio.run(service.send_message(user_id="user-1", session_id="chat-1", query="what is RAG"))

    assert sessions.renames == ["what is RAG"]


def test_send_message_leaves_an_already_named_chat_alone() -> None:
    service, sessions, _, _ = _service(title="My research")

    asyncio.run(service.send_message(user_id="user-1", session_id="chat-1", query="what is RAG"))

    assert sessions.renames == []


def test_history_excludes_the_current_turn_and_grows_across_turns() -> None:
    service, _, _, graph = _service()

    asyncio.run(service.send_message(user_id="user-1", session_id="chat-1", query="first q"))
    asyncio.run(service.send_message(user_id="user-1", session_id="chat-1", query="second q"))

    assert graph.histories[0] == ()  # the first turn has no prior context
    second = graph.histories[1]
    assert [(t.role, t.content) for t in second] == [
        ("user", "first q"),
        ("assistant", "reply: first q"),
    ]


def test_history_is_windowed_to_the_configured_maximum() -> None:
    service, _, _, graph = _service(history_max=2)

    for n in range(3):
        asyncio.run(service.send_message(user_id="user-1", session_id="chat-1", query=f"q{n}"))

    assert len(graph.histories[2]) == 2  # only the two most recent messages


# --- streaming path ----------------------------------------------------------------------------


def test_stream_message_forwards_steps_and_tokens_then_persists() -> None:
    step = Step(id="retrieve:1", node="retrieve", status="running", label="Searching")
    graph = FakeAnswerer(
        events=(step, Token("Bharat"), Token(" is India."), Done(
            GeneratedAnswer(answer="Bharat is India.", language="en", citations=())
        ))
    )
    service, _, messages, _ = _service(answerer=graph)

    events = asyncio.run(
        _collect(service.stream_message(user_id="user-1", session_id="chat-1", query="q"))
    )

    assert isinstance(events[0], StepChunk)
    assert events[0].step.label == "Searching"
    assert [e.text for e in events if isinstance(e, TokenChunk)] == ["Bharat", " is India."]
    completed = [e for e in events if isinstance(e, CompletedMessage)]
    assert completed[0].message.content == "Bharat is India."
    # Steps are ephemeral — only the two turns are persisted, never the step.
    assert [m.role for m in messages.messages] == ["user", "assistant"]


def test_stream_message_auto_titles_a_fresh_chat() -> None:
    service, sessions, _, _ = _service()

    asyncio.run(
        _collect(service.stream_message(user_id="user-1", session_id="chat-1", query="hello there"))
    )

    assert sessions.renames == ["hello there"]
