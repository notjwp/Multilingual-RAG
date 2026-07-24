"""Chat-scoped document routes (M18): upload/list/delete are scoped to one chat.

Uses the ``app.state`` injection seam — a fake chat service (for ownership) and a fake document
service — so no database is needed.
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    ChatSessionRecord,
    DocumentMetadata,
    DocumentRecord,
    IngestionJobRecord,
    MessageRecord,
    UserRecord,
)


class FakeChatService:
    """Owns "chat-1" for user-1; ``get_session`` raises for anything else (ownership 404)."""

    def __init__(self, owned: tuple[str, ...] = ("chat-1",)) -> None:
        self.owned = owned

    async def get_session(
        self, *, user_id: str, session_id: str
    ) -> tuple[ChatSessionRecord, tuple[MessageRecord, ...]]:
        if user_id != "user-1" or session_id not in self.owned:
            raise AppError(
                f"Chat session not found: {session_id}",
                code="chat_not_found",
                status_code=404,
            )
        record = ChatSessionRecord(
            session_id=session_id,
            user_id=user_id,
            title="A chat",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return record, ()


class FakeDocumentService:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, str | None, Path]] = []  # (user_id, session_id, path)
        self.listed_sessions: list[str | None] = []
        self.deleted: list[tuple[str, str | None]] = []  # (document_id, session_id)
        self.record = DocumentRecord(
            document=DocumentMetadata(
                document_id="doc-1",
                source="sample.txt",
                content_type="text/plain",
                checksum="abc123",
                language="en",
            ),
            chunk_count=3,
        )
        self.job = IngestionJobRecord(
            job_id="job-1",
            user_id="user-1",
            file_path="sample.txt",
            status="queued",
        )

    async def create_ingestion_job(
        self, *, user_id: str, session_id: str | None = None, path: Path
    ) -> IngestionJobRecord:
        assert user_id == "user-1"
        self.jobs.append((user_id, session_id, path))
        return self.job

    async def get_ingestion_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        assert user_id == "user-1"
        assert job_id == "job-1"
        return self.job

    async def list_documents(
        self, *, user_id: str, session_id: str | None = None
    ) -> tuple[DocumentRecord, ...]:
        assert user_id == "user-1"
        self.listed_sessions.append(session_id)
        return (self.record,)

    async def delete_document(
        self, *, user_id: str, document_id: str, session_id: str | None = None
    ) -> DocumentRecord:
        assert user_id == "user-1"
        self.deleted.append((document_id, session_id))
        return self.record


def _app(tmp_path: Path) -> tuple[object, FakeDocumentService]:
    app = create_app(Settings(environment="test", raw_document_directory=tmp_path / "raw"))
    docs = FakeDocumentService()
    app.state.document_service = docs
    app.state.chat_service = FakeChatService()
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    return app, docs


def test_upload_into_a_chat_scopes_the_job_to_that_chat(tmp_path: Path) -> None:
    app, docs = _app(tmp_path)
    enqueued: list[str] = []
    app.state.enqueue_ingestion = enqueued.append  # type: ignore[attr-defined]

    with TestClient(app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/v1/chats/chat-1/documents",
            files={"file": ("sample.txt", b"hello world", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert response.json()["status"] == "queued"
    # The job was scoped to chat-1, the file was saved to disk, and Celery was enqueued.
    assert docs.jobs[0][1] == "chat-1"
    assert docs.jobs[0][2].exists()
    assert enqueued == ["job-1"]


def test_list_and_delete_are_chat_scoped(tmp_path: Path) -> None:
    app, docs = _app(tmp_path)

    with TestClient(app) as client:  # type: ignore[arg-type]
        list_response = client.get("/v1/chats/chat-1/documents")
        delete_response = client.delete("/v1/chats/chat-1/documents/doc-1")
        job_response = client.get("/v1/ingestion-jobs/job-1")

    assert list_response.status_code == 200
    assert delete_response.status_code == 200
    assert job_response.status_code == 200
    assert docs.listed_sessions == ["chat-1"]  # listing was scoped to the chat
    assert docs.deleted == [("doc-1", "chat-1")]  # delete was scoped to the chat


def test_operating_on_a_chat_you_do_not_own_is_rejected(tmp_path: Path) -> None:
    app, docs = _app(tmp_path)

    with TestClient(app) as client:  # type: ignore[arg-type]
        upload = client.post(
            "/v1/chats/not-mine/documents",
            files={"file": ("sample.txt", b"hello", "text/plain")},
        )
        listed = client.get("/v1/chats/not-mine/documents")

    assert upload.status_code == 404
    assert upload.json()["error"] == "chat_not_found"
    assert listed.status_code == 404
    assert docs.jobs == []  # ownership rejected before any job was created
