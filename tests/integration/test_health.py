from fastapi import APIRouter
from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError


def test_healthz_returns_service_status() -> None:
    app = create_app(Settings(environment="test", app_name="test-rag"))

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "test-rag",
        "environment": "test",
    }


def test_readyz_reports_missing_openai_key() -> None:
    app = create_app(Settings(environment="test", openai_api_key=None))

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["checks"] == {
        "configuration": True,
        "openai_api_key_configured": False,
    }


def test_readyz_reports_configured_openai_key() -> None:
    app = create_app(Settings(environment="test", openai_api_key="test-key"))

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["checks"]["openai_api_key_configured"] is True


def test_app_error_handler_returns_standard_error_response() -> None:
    app = create_app(Settings(environment="test"))
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> None:
        raise AppError("Expected failure.", code="expected_failure", status_code=418)

    app.include_router(router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 418
    assert response.json() == {
        "error": "expected_failure",
        "message": "Expected failure.",
        "details": None,
    }

