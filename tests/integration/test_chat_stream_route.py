"""Streaming chat route tests — fake ChatService on app.state (no DB, no model, no network)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.chat.service import ChatStreamEvent, CompletedMessage, TokenChunk
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    AnswerCitation,
    ChatSessionRecord,
    MessageRecord,
    UserRecord,
)

_NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _session(session_id: str = "chat-1", title: str = "New chat") -> ChatSessionRecord:
    return ChatSessionRecord(session_id=session_id, user_id="user-1", title=title, created_at=_NOW)


def _message(role: str, content: str, citations: tuple[AnswerCitation, ...] = ()) -> MessageRecord:
    return MessageRecord(
        message_id=f"msg-{role}", session_id="chat-1", role=role, content=content,
        created_at=_NOW, citations=citations,
    )


class FakeStreamingChatService:
    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        if session_id == "missing":
            raise AppError(
                f"Chat session not found: {session_id}", code="chat_not_found", status_code=404
            )
        return _session(session_id), ()

    async def stream_message(
        self, *, user_id: str, session_id: str, query: str
    ) -> AsyncIterator[ChatStreamEvent]:
        yield TokenChunk("Bhara")
        yield TokenChunk("t is a nation.")
        citation = AnswerCitation(
            chunk_id="c1", document_id="d1", source="doc.txt", page=2, text="snippet"
        )
        yield CompletedMessage(_message("assistant", "Bharat is a nation.", (citation,)))


def _authed_app() -> object:
    app = create_app(Settings(environment="test"))
    app.state.chat_service = FakeStreamingChatService()
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    return app


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    """Return (event, data) pairs from an SSE payload (event defaults to 'message')."""
    events: list[tuple[str, dict[str, object]]] = []
    for frame in body.strip().split("\n\n"):
        event = "message"
        data = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        events.append((event, json.loads(data)))
    return events


def test_stream_emits_tokens_then_done() -> None:
    app = _authed_app()
    with TestClient(app) as client:
        response = client.post("/v1/chats/chat-1/messages/stream", json={"query": "bharat"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)

    assert [data["token"] for event, data in events if event == "message"] == [
        "Bhara",
        "t is a nation.",
    ]
    done = [data for event, data in events if event == "done"]
    assert len(done) == 1
    assert done[0]["message_id"] == "msg-assistant"
    assert done[0]["citations"][0]["text"] == "snippet"  # type: ignore[index]


def test_stream_unknown_chat_returns_404() -> None:
    app = _authed_app()
    with TestClient(app) as client:
        response = client.post("/v1/chats/missing/messages/stream", json={"query": "x"})

    # Ownership is checked before streaming starts, so this is a clean 404 JSON, not an SSE stream.
    assert response.status_code == 404
    assert response.json()["error"] == "chat_not_found"


def test_stream_requires_authentication() -> None:
    app = create_app(Settings(environment="test"))
    app.state.chat_service = FakeStreamingChatService()
    with TestClient(app) as client:
        response = client.post("/v1/chats/chat-1/messages/stream", json={"query": "x"})
    assert response.status_code == 401
