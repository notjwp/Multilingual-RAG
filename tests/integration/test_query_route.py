from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.api.routes.query import QueryRequest, QueryResponse, RagQueryService
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import UserRecord


class FakeQueryService:
    def __init__(self) -> None:
        self.requests: list[tuple[QueryRequest, str]] = []

    def answer_query(self, request: QueryRequest, *, user_id: str) -> QueryResponse:
        self.requests.append((request, user_id))
        return QueryResponse(
            answer="Test answer",
            language=request.preferred_language or "en",
            query_language="en",
            citations=(),
            retrieved_chunks=(),
        )


def _authed_app() -> tuple[object, FakeQueryService]:
    app = create_app(Settings(environment="test"))
    service = FakeQueryService()
    app.state.query_service = service
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    return app, service


def test_query_route_authenticates_and_passes_user_id() -> None:
    app, service = _authed_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "What is RAG?",
                "preferred_language": "fr",
                "top_k": 3,
                "filters": {"language": "en"},
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"
    assert response.json()["language"] == "fr"
    request, user_id = service.requests[0]
    assert request.filters == {"language": "en"}
    # The authenticated user's id must reach the query service.
    assert user_id == "user-1"


def test_query_route_requires_authentication() -> None:
    # No app.state.current_user override and no bearer token -> must be rejected.
    app = create_app(Settings(environment="test"))
    app.state.query_service = FakeQueryService()

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "What is RAG?"})

    assert response.status_code == 401
    assert response.json()["error"] == "authentication_required"


def test_query_route_rejects_reserved_user_id_filter() -> None:
    # A client must not be able to smuggle a user_id filter to reach another tenant.
    app = create_app(Settings(environment="test"))
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")

    # The reserved-filter guard fires before any retrieval, so these must never be called.
    class _Boom:
        def retrieve(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("retrieval must not run for a rejected filter")

    class _Gen:
        def generate_answer(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("generation must not run for a rejected filter")

    app.state.query_service = RagQueryService(
        retrieval_service=_Boom(),  # type: ignore[arg-type]
        answer_generator=_Gen(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "What is RAG?", "filters": {"user_id": "someone-else"}},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "reserved_filter_key"
