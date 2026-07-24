"""Auth route: token refresh + security headers (fake user on app.state, no DB)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import UserRecord


def test_refresh_returns_a_fresh_token() -> None:
    app = create_app(Settings(environment="test"))
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    with TestClient(app) as client:
        response = client.post("/v1/auth/refresh")

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]  # a signed JWT string
    assert body["user"]["email"] == "user@example.com"


def test_refresh_requires_authentication() -> None:
    app = create_app(Settings(environment="test"))
    with TestClient(app) as client:
        response = client.post("/v1/auth/refresh")
    assert response.status_code == 401


def test_security_headers_present() -> None:
    app = create_app(Settings(environment="test"))
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    # HSTS is prod/staging only.
    assert "strict-transport-security" not in response.headers
