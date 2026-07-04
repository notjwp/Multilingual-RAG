from fastapi.testclient import TestClient

from multilingual_rag.api.app import create_app
from multilingual_rag.api.routes.query import QueryRequest, QueryResponse
from multilingual_rag.core.config import Settings


class FakeQueryService:
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        self.requests.append(request)
        return QueryResponse(
            answer="Test answer",
            language=request.preferred_language or "en",
            query_language="en",
            citations=(),
            retrieved_chunks=(),
        )


def test_query_route_uses_injected_query_service() -> None:
    app = create_app(Settings(environment="test"))
    service = FakeQueryService()
    app.state.query_service = service

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
    assert service.requests[0].filters == {"language": "en"}

