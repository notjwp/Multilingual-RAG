"""Chat route tests — fake ChatService on app.state (no DB, no model)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.core.config import Settings
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


class FakeChatService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def create_session(self, *, user_id: str, title: str | None = None) -> ChatSessionRecord:
        self.calls.append(("create", {"user_id": user_id, "title": title}))
        return _session(title=title or "New chat")

    async def list_sessions(self, *, user_id: str) -> tuple[ChatSessionRecord, ...]:
        return (_session("chat-1", "First"), _session("chat-2", "Second"))

    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        return _session(session_id), (_message("user", "hi"), _message("assistant", "hello"))

    async def rename_session(
        self, *, user_id: str, session_id: str, title: str
    ) -> ChatSessionRecord:
        return _session(session_id, title)

    async def delete_session(self, *, user_id: str, session_id: str) -> ChatSessionRecord:
        return _session(session_id)

    async def send_message(self, *, user_id: str, session_id: str, query: str) -> MessageRecord:
        self.calls.append(("send", {"user_id": user_id, "session_id": session_id, "query": query}))
        citation = AnswerCitation(
            chunk_id="c1", document_id="d1", source="doc.txt", page=2, text="snippet"
        )
        return _message("assistant", "the answer", (citation,))


def _authed_app() -> tuple[object, FakeChatService]:
    app = create_app(Settings(environment="test"))
    service = FakeChatService()
    app.state.chat_service = service
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    return app, service


def test_create_list_and_get_chat() -> None:
    app, _ = _authed_app()
    with TestClient(app) as client:
        created = client.post("/v1/chats", json={"title": "Docs"})
        assert created.status_code == 201
        assert created.json()["title"] == "Docs"

        listed = client.get("/v1/chats")
        assert [c["title"] for c in listed.json()] == ["First", "Second"]

        detail = client.get("/v1/chats/chat-1")
        assert detail.status_code == 200
        assert [m["role"] for m in detail.json()["messages"]] == ["user", "assistant"]


def test_send_message_returns_assistant_with_citations() -> None:
    app, service = _authed_app()
    with TestClient(app) as client:
        response = client.post("/v1/chats/chat-1/messages", json={"query": "bharat kya hai"})

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert body["content"] == "the answer"
    assert body["citations"][0]["text"] == "snippet"
    assert service.calls[-1] == ("send", {"user_id": "user-1", "session_id": "chat-1",
                                          "query": "bharat kya hai"})


def test_rename_and_delete_chat() -> None:
    app, _ = _authed_app()
    with TestClient(app) as client:
        renamed = client.patch("/v1/chats/chat-1", json={"title": "Renamed"})
        assert renamed.json()["title"] == "Renamed"
        deleted = client.delete("/v1/chats/chat-1")
        assert deleted.status_code == 200


def test_chat_routes_require_authentication() -> None:
    app = create_app(Settings(environment="test"))
    app.state.chat_service = FakeChatService()
    with TestClient(app) as client:
        assert client.get("/v1/chats").status_code == 401
        assert client.post("/v1/chats", json={}).status_code == 401
