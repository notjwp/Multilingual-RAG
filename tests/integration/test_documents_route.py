from pathlib import Path

from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    DocumentMetadata,
    DocumentRecord,
    IngestionJobRecord,
    UserRecord,
)


class FakeDocumentService:
    def __init__(self) -> None:
        self.indexed_paths: list[Path] = []
        self.deleted_document_ids: list[str] = []
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

    async def create_ingestion_job(self, *, user_id: str, path: Path) -> IngestionJobRecord:
        assert user_id == "user-1"
        self.indexed_paths.append(path)
        return self.job

    async def get_ingestion_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord:
        assert user_id == "user-1"
        assert job_id == "job-1"
        return self.job

    async def list_documents(self, *, user_id: str) -> tuple[DocumentRecord, ...]:
        assert user_id == "user-1"
        return (self.record,)

    async def get_document(self, *, user_id: str, document_id: str) -> DocumentRecord:
        assert user_id == "user-1"
        assert document_id == "doc-1"
        return self.record

    async def delete_document(self, *, user_id: str, document_id: str) -> DocumentRecord:
        assert user_id == "user-1"
        self.deleted_document_ids.append(document_id)
        return self.record


def test_upload_document_route_saves_upload_and_uses_injected_service(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            raw_document_directory=tmp_path / "raw",
        )
    )
    service = FakeDocumentService()
    app.state.document_service = service
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    enqueued_jobs: list[str] = []
    app.state.enqueue_ingestion = enqueued_jobs.append

    with TestClient(app) as client:
        response = client.post(
            "/v1/documents/upload",
            files={"file": ("sample.txt", b"hello world", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert response.json()["status"] == "queued"
    assert service.indexed_paths[0].exists()
    assert enqueued_jobs == ["job-1"]


def test_get_and_delete_document_routes_use_injected_service(tmp_path: Path) -> None:
    app = create_app(Settings(environment="test", raw_document_directory=tmp_path / "raw"))
    service = FakeDocumentService()
    app.state.document_service = service
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")

    with TestClient(app) as client:
        list_response = client.get("/v1/documents")
        get_response = client.get("/v1/documents/doc-1")
        delete_response = client.delete("/v1/documents/doc-1")
        job_response = client.get("/v1/ingestion-jobs/job-1")

    assert list_response.status_code == 200
    assert get_response.status_code == 200
    assert delete_response.status_code == 200
    assert job_response.status_code == 200
    assert service.deleted_document_ids == ["doc-1"]
