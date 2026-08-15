"""Agent graph unit tests — fake retriever, stream client, and grader. No network, no model.

The first three tests are the invariants the old StreamingAnswerGenerator asserted, carried over
so the orchestration rewrite cannot silently lose them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from openai import RateLimitError

from multilingual_rag.agent.events import AgentEvent, Done, Step, Token
from multilingual_rag.agent.factory import build_rag_graph
from multilingual_rag.agent.grading.base import Grade
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    ConversationTurn,
    RetrievalContext,
    VectorSearchResult,
)
from multilingual_rag.retrieval.routing import LanguageRoute
from multilingual_rag.vectorstores.base import VectorFilter

NATIVE = "भारत की राजधानी क्या है"


def _http_response(status_code: int) -> httpx.Response:
    """A minimal response object, since the OpenAI SDK's errors require one."""
    return httpx.Response(status_code, request=httpx.Request("POST", "http://test/v1"))


def _result(chunk_id: str = "doc-1:0", score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id, document_id="doc-1", text="Bharat is India.", language="en",
        source="s.txt", chunk_index=0, score=score, page=1, token_count=4, metadata={},
    )


def _context(query: str = "what is bharat", results: tuple[VectorSearchResult, ...] | None = None):
    return RetrievalContext(
        query=query,
        query_language="en",
        results=(_result(),) if results is None else results,
    )


class FakeStreamClient:
    def __init__(self, deltas: Sequence[str], *, rewritten: str = "REWRITTEN") -> None:
        self._deltas = deltas
        self._rewritten = rewritten
        self.acomplete_calls = 0
        self.systems: list[str] = []
        self.stream_history: tuple[object, ...] = ()
        self.stream_prompts: list[str] = []

    async def astream_completion(
        self, *, model: str, system: str, prompt: str, history: Sequence[ConversationTurn] = ()
    ) -> AsyncIterator[str]:
        self.stream_history = tuple(history)
        self.stream_prompts.append(prompt)
        self.systems.append(system)
        for delta in self._deltas:
            yield delta

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        self.acomplete_calls += 1
        return self._rewritten


class FakeRetriever:
    """Records every route/retrieve call; returns scripted contexts, one per attempt."""

    def __init__(
        self,
        contexts: Sequence[RetrievalContext] | None = None,
        *,
        assert_off_main_thread: bool = False,
    ) -> None:
        self._contexts = list(contexts) if contexts else [_context()]
        self._assert_off_main_thread = assert_off_main_thread
        self.queries: list[str] = []
        self.calls: list[tuple[str, str | None]] = []  # (user_id, session_id) per retrieval
        self.routes: list[LanguageRoute] = []
        self.route_kwargs: list[tuple[str | None, bool]] = []

    def route(
        self,
        query: str,
        *,
        force_language: str | None = None,
        skip_transliteration: bool = False,
    ) -> LanguageRoute:
        if self._assert_off_main_thread:
            # detect_target_language calls asyncio.run internally on the google-detector path,
            # which explodes on a running event loop. Routing must be offloaded.
            assert threading.current_thread() is not threading.main_thread()
        self.route_kwargs.append((force_language, skip_transliteration))
        if skip_transliteration:
            route = LanguageRoute(query, None, None)
        elif force_language is not None:
            route = LanguageRoute(f"{force_language}:{query}", force_language, f"x:{query}")
        else:
            route = LanguageRoute(NATIVE, "hi", NATIVE)
        self.routes.append(route)
        return route

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
        route: LanguageRoute | None = None,
    ) -> RetrievalContext:
        self.queries.append(query)
        self.calls.append((user_id, session_id))
        index = min(len(self.queries) - 1, len(self._contexts) - 1)
        return self._contexts[index]


class FakeGrader:
    """Returns scripted verdicts in order; repeats the last one once exhausted."""

    def __init__(self, *verdicts: bool) -> None:
        self._verdicts = list(verdicts) or [True]
        self.calls = 0

    async def grade(self, *, query: str, results: Sequence[VectorSearchResult]) -> Grade:
        index = min(self.calls, len(self._verdicts) - 1)
        self.calls += 1
        relevant = self._verdicts[index]
        return Grade(relevant=relevant, reason="scripted", top_score=0.9 if relevant else 0.1)


def _graph(
    *,
    retriever: FakeRetriever | None = None,
    client: FakeStreamClient | None = None,
    grader: FakeGrader | None = None,
    settings: Settings | None = None,
):
    return build_rag_graph(
        settings or Settings(environment="test"),
        retriever=retriever or FakeRetriever(),
        client=client or FakeStreamClient(["Bharat [1]."]),
        grader=grader or FakeGrader(True),
    )


async def _collect(generator: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in generator]


def _steps(events: Sequence[AgentEvent]) -> list[tuple[str, str]]:
    return [(e.node, e.status) for e in events if isinstance(e, Step)]


# --- ported invariants -------------------------------------------------------------------------


def test_stream_yields_tokens_then_grounded_done() -> None:
    graph = _graph(client=FakeStreamClient(["Bharat", " is ", "India [1]."]))

    events = asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    tokens = [e for e in events if isinstance(e, Token)]
    done = [e for e in events if isinstance(e, Done)]
    assert [t.text for t in tokens] == ["Bharat", " is ", "India [1]."]
    assert len(done) == 1
    assert done[0].answer.answer == "Bharat is India [1]."
    assert done[0].answer.citations[0].chunk_id == "doc-1:0"  # [1] -> results[0]


def test_first_turn_skips_condense_and_searches_the_raw_query() -> None:
    retriever, client = FakeRetriever(), FakeStreamClient(["Bharat [1]."])
    graph = _graph(retriever=retriever, client=client)

    asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    assert client.acomplete_calls == 0  # the conditional entry edge skipped condense
    assert retriever.queries == ["what is bharat"]
    assert client.stream_history == ()


def test_follow_up_condenses_then_retrieves_the_rewritten_query() -> None:
    retriever = FakeRetriever()
    client = FakeStreamClient(["Wexler [1]."], rewritten="Who founded the Zorblax Protocol?")
    graph = _graph(retriever=retriever, client=client)
    history = (
        ConversationTurn(role="user", content="Tell me about the Zorblax Protocol"),
        ConversationTurn(role="assistant", content="It is a fictional standard."),
    )

    asyncio.run(_collect(graph.stream("who founded it?", user_id="user-1", history=history)))

    assert client.acomplete_calls == 1
    assert retriever.queries == ["Who founded the Zorblax Protocol?"]
    assert client.stream_history == history  # generation still sees the prior turns


# --- the agentic behaviour ---------------------------------------------------------------------


def test_weak_retrieval_repairs_and_retries_with_a_different_route() -> None:
    retriever = FakeRetriever([_context(results=()), _context()])
    graph = _graph(retriever=retriever, grader=FakeGrader(False, True))

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert len(retriever.queries) == 2  # the cycle ran exactly once
    # First route transliterated; the repair retried the raw form instead.
    assert retriever.route_kwargs == [(None, False), (None, True)]
    assert ("repair", "done") in _steps(events)
    assert [e for e in events if isinstance(e, Done)]  # still answered


def test_exhausted_repairs_answer_without_context_and_cite_nothing() -> None:
    retriever = FakeRetriever([_context(results=())])
    client = FakeStreamClient(["Your documents don't cover that."])
    graph = _graph(retriever=retriever, client=client, grader=FakeGrader(False))

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert len(retriever.queries) == 2  # one repair (agent_max_repairs default 1), then stop
    done = [e for e in events if isinstance(e, Done)]
    assert done[0].answer.citations == ()  # the give-up path cannot cite


def test_max_repairs_zero_disables_the_cycle_entirely() -> None:
    retriever = FakeRetriever([_context(results=())])
    graph = _graph(
        retriever=retriever,
        grader=FakeGrader(False),
        settings=Settings(environment="test", agent_max_repairs=0),
    )

    asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    assert len(retriever.queries) == 1


def test_user_and_session_id_reach_every_retrieval_including_after_repair() -> None:
    # The Phase A tenancy guarantee, made explicit. Retrieval is a node rather than an
    # LLM-callable tool precisely so the model can never author these.
    retriever = FakeRetriever([_context(results=()), _context()])
    graph = _graph(retriever=retriever, grader=FakeGrader(False, True))

    asyncio.run(
        _collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-7", session_id="chat-3"))
    )

    assert retriever.calls == [("user-7", "chat-3"), ("user-7", "chat-3")]


def test_language_routing_runs_off_the_event_loop() -> None:
    # detect_target_language calls asyncio.run internally (google detector). If route_language
    # ran on the loop this would be a production-only RuntimeError — the default word-list
    # detector never takes that path, so the suite would stay green.
    retriever = FakeRetriever(assert_off_main_thread=True)
    graph = _graph(retriever=retriever)

    asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    assert retriever.routes  # the assertion inside route() actually ran


# --- streamed steps ----------------------------------------------------------------------------


def test_step_events_describe_a_happy_path() -> None:
    graph = _graph()

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert _steps(events) == [
        ("route_language", "done"),
        ("retrieve", "running"),
        ("retrieve", "done"),
        ("generate", "running"),
        ("generate", "done"),
    ]


def test_step_events_describe_a_repair_path() -> None:
    graph = _graph(
        retriever=FakeRetriever([_context(results=()), _context()]),
        grader=FakeGrader(False, True),
    )

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert _steps(events) == [
        ("route_language", "done"),
        ("retrieve", "running"),
        ("retrieve", "done"),
        ("repair", "running"),
        ("repair", "done"),
        ("retrieve", "running"),
        ("retrieve", "done"),
        ("generate", "running"),
        ("generate", "done"),
    ]


def test_running_and_done_steps_share_an_id_so_the_ui_can_upsert() -> None:
    graph = _graph()

    events = asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    retrieve_steps = [e for e in events if isinstance(e, Step) and e.node == "retrieve"]
    assert {s.id for s in retrieve_steps} == {"retrieve:1"}


def test_an_english_query_reports_no_language_re_routing() -> None:
    class PlainRetriever(FakeRetriever):
        def route(self, query, *, force_language=None, skip_transliteration=False):
            return LanguageRoute(query, None, None)

    graph = _graph(retriever=PlainRetriever())

    events = asyncio.run(_collect(graph.stream("what is the capital", user_id="user-1")))

    assert not [e for e in events if isinstance(e, Step) and e.node == "route_language"]


# --- blocking path -----------------------------------------------------------------------------


def test_answer_returns_the_result_without_any_stream_subscriber() -> None:
    # Under ainvoke, emit() is a no-op — the same generate node just accumulates instead of
    # streaming. This is why generation no longer needs a blocking twin.
    graph = _graph(client=FakeStreamClient(["Bharat is India [1]."]))

    result = asyncio.run(graph.answer("what is bharat", user_id="user-1"))

    assert result.answer.answer == "Bharat is India [1]."
    assert result.context.results[0].chunk_id == "doc-1:0"


def test_a_provider_error_surfaces_as_an_apperror_through_the_stream() -> None:
    # chat_stream.py only renders `event: error` for AppError. If a raw OpenAIError escaped
    # astream instead, the SSE response would truncate mid-answer with no error frame and the
    # UI would hang on a half-written bubble.
    class ExplodingClient(FakeStreamClient):
        async def astream_completion(self, *, model, system, prompt, history=()):
            raise RateLimitError("slow down", response=_http_response(429), body=None)
            yield ""  # pragma: no cover — unreachable, makes this an async generator

    graph = _graph(client=ExplodingClient([]))

    with pytest.raises(AppError) as caught:
        asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    assert caught.value.code == "generation_rate_limited"
    assert caught.value.status_code == 429


def test_answer_preserves_the_users_wording_after_a_condense() -> None:
    retriever = FakeRetriever()
    client = FakeStreamClient(["Wexler [1]."], rewritten="Who founded the Zorblax Protocol?")
    graph = _graph(retriever=retriever, client=client)
    history = (ConversationTurn(role="user", content="Tell me about Zorblax"),)

    result = asyncio.run(graph.answer("who founded it?", user_id="user-1", history=history))

    assert retriever.queries == ["Who founded the Zorblax Protocol?"]  # searched the rewrite
    assert result.context.query == "who founded it?"  # reported the user's actual words
