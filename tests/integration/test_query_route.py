from collections.abc import Sequence

from fastapi.testclient import TestClient

from multilingual_rag.agent.state import AgentResult
from multilingual_rag.api.app import create_app
from multilingual_rag.api.routes.query import query_response_from_models
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import (
    ConversationTurn,
    GeneratedAnswer,
    RetrievalContext,
    UserRecord,
)
from multilingual_rag.vectorstores.base import VectorFilter


class FakeRagGraph:
    """Stands in for RagGraph on app.state — records what the route passed down."""

    def __init__(self, *, answer: str = "Test answer", language: str = "en") -> None:
        self._answer = answer
        self._language = language
        self.calls: list[tuple[str, str, int | None, VectorFilter | None, str | None]] = []

    async def answer(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AgentResult:
        self.calls.append((query, user_id, top_k, filters, preferred_language))
        return AgentResult(
            answer=GeneratedAnswer(
                answer=self._answer,
                language=preferred_language or self._language,
                citations=(),
            ),
            context=RetrievalContext(query=query, query_language="en", results=()),
        )


def _authed_app() -> tuple[object, FakeRagGraph]:
    app = create_app(Settings(environment="test"))
    graph = FakeRagGraph()
    app.state.rag_graph = graph
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")
    return app, graph


def test_query_route_authenticates_and_passes_user_id() -> None:
    app, graph = _authed_app()

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
    query, user_id, top_k, filters, preferred = graph.calls[0]
    assert (query, top_k, filters, preferred) == ("What is RAG?", 3, {"language": "en"}, "fr")
    # The authenticated user's id must reach the graph.
    assert user_id == "user-1"


def test_response_surfaces_transliteration_fields() -> None:
    # The response mapper must carry RetrievalContext's transliteration transparency through.
    context = RetrievalContext(
        query="bharat kya hai",
        query_language="en",
        results=(),
        transliterated_query="भारत क्या है",
        transliteration_applied=True,
    )
    answer = GeneratedAnswer(answer="ans", language="hi", citations=())

    response = query_response_from_models(answer, context)

    assert response.transliteration_applied is True
    assert response.transliterated_query == "भारत क्या है"


def test_query_route_requires_authentication() -> None:
    # No app.state.current_user override and no bearer token -> must be rejected.
    app = create_app(Settings(environment="test"))
    app.state.rag_graph = FakeRagGraph()

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "What is RAG?"})

    assert response.status_code == 401
    assert response.json()["error"] == "authentication_required"


def test_query_route_rejects_reserved_user_id_filter() -> None:
    # A client must not be able to smuggle a user_id filter to reach another tenant.
    app = create_app(Settings(environment="test"))
    app.state.current_user = UserRecord(user_id="user-1", email="user@example.com")

    class _Boom:
        """The guard fires before the graph runs, so this must never be invoked."""

        async def answer(self, *args: object, **kwargs: object) -> AgentResult:
            raise AssertionError("the graph must not run for a rejected filter")

    app.state.rag_graph = _Boom()

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "What is RAG?", "filters": {"user_id": "someone-else"}},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "reserved_filter_key"
