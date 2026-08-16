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
from multilingual_rag.agent.grounding.factory import build_grounding_judge
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    ConversationTurn,
    RetrievalContext,
    VectorSearchResult,
)
from multilingual_rag.evaluation.llm_judge import LlmFaithfulnessJudge
from multilingual_rag.generation.prompts import NO_CONTEXT_SYSTEM
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
        # (force_language, skip_transliteration) per route decision
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


def test_weak_retrieval_repairs_and_retries() -> None:
    retriever = FakeRetriever([_context(results=()), _context()])
    graph = _graph(retriever=retriever, grader=FakeGrader(False, True))

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert len(retriever.queries) == 2  # the cycle ran exactly once
    assert ("repair", "done") in _steps(events)
    assert [e for e in events if isinstance(e, Done)]  # still answered


def test_the_first_repair_is_not_a_retreat_to_the_raw_query() -> None:
    # raw_fallback is the last resort, never the opener: leading with it measured 0.767 against
    # 0.800 for no agent at all, because raw romanized recall is 0.133. See agent/repair.py.
    retriever = FakeRetriever([_context(results=()), _context()])
    graph = _graph(retriever=retriever, grader=FakeGrader(False, True))

    asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert all(not skip_transliteration for _, skip_transliteration in retriever.route_kwargs)


def test_exhausted_repairs_answer_without_context_and_cite_nothing() -> None:
    retriever = FakeRetriever([_context(results=())])
    client = FakeStreamClient(["Your documents don't cover that."])
    graph = _graph(retriever=retriever, client=client, grader=FakeGrader(False))

    events = asyncio.run(_collect(graph.stream("bharat ki rajdhani kya hai", user_id="user-1")))

    assert len(retriever.queries) == 2  # one repair (agent_max_repairs default 1), then stop
    done = [e for e in events if isinstance(e, Done)]
    assert done[0].answer.citations == ()  # the give-up path cannot cite


def test_a_repair_that_retrieves_worse_does_not_lose_the_better_attempt() -> None:
    # The regression this pins: a repair is a bet and it can lose. Falling back to the raw
    # romanized query is right only when the transliterated search genuinely failed, and the score
    # bands overlap, so the grader mislabels some correct retrievals as weak. Measured on XQuAD-hi
    # before this existed: agentic recall@5 0.733 vs 0.800 for the plain pipeline. Generation must
    # answer from the best attempt, never merely the last.
    good = _context(results=(_result("good:0", score=0.44),))
    worse = _context(results=(_result("worse:0", score=0.05),))
    graph = _graph(
        retriever=FakeRetriever([good, worse]),
        # Both attempts graded weak, but the first scored far higher.
        grader=FakeGrader(False, False),
        settings=Settings(environment="test", agent_max_repairs=1),
    )

    result = asyncio.run(graph.answer("bharat ki rajdhani kya hai", user_id="user-1"))

    assert result.context.results[0].chunk_id == "good:0"


def test_a_tie_keeps_the_original_attempt() -> None:
    # The subtle case the two neighbours don't cover. When both attempts grade the same, the
    # incumbent wins — deliberately *not* "higher score wins", because ranking a transliterated
    # search against a raw one by cosine is the relative judgement transliteration/detect.py
    # records as unreliable, and it measured 0.750 vs 0.800 when tried.
    first = _context(results=(_result("first:0", score=0.20),))
    second = _context(results=(_result("second:0", score=0.99),))  # higher score, same verdict
    graph = _graph(
        retriever=FakeRetriever([first, second]),
        grader=FakeGrader(False, False),
        settings=Settings(environment="test", agent_max_repairs=1),
    )

    result = asyncio.run(graph.answer("bharat ki rajdhani kya hai", user_id="user-1"))

    assert result.context.results[0].chunk_id == "first:0"


def test_a_repair_that_retrieves_better_is_adopted() -> None:
    weak = _context(results=(_result("weak:0", score=0.10),))
    strong = _context(results=(_result("strong:0", score=0.90),))
    graph = _graph(
        retriever=FakeRetriever([weak, strong]),
        grader=FakeGrader(False, True),
        settings=Settings(environment="test", agent_max_repairs=1),
    )

    result = asyncio.run(graph.answer("bharat ki rajdhani kya hai", user_id="user-1"))

    assert result.context.results[0].chunk_id == "strong:0"


def test_the_answer_language_follows_the_router_not_langdetect() -> None:
    """Found by a live transcript: a no-context refusal for `bharat ki rajdhani kya hai` came back
    in Albanian, because langdetect labels that romanized Hindi as Swahili and
    resolve_answer_language trusted it. route_language had already identified `hi` correctly with
    the purpose-built detector. The normal path masks this — Devanagari passages in the prompt make
    the model mirror them regardless — so only the empty-retrieval path exposed it."""
    misdetected = RetrievalContext(query="bharat ki rajdhani kya hai", query_language="sw",
                                   results=())
    graph = _graph(
        retriever=FakeRetriever([misdetected]),  # FakeRetriever routes to hi
        grader=FakeGrader(False),
        settings=Settings(environment="test", agent_max_repairs=0),
    )

    result = asyncio.run(graph.answer("bharat ki rajdhani kya hai", user_id="user-1"))

    assert result.answer.language == "hi", "the router's verdict must beat langdetect's guess"


def test_an_explicit_preferred_language_still_wins() -> None:
    misdetected = RetrievalContext(query="q", query_language="sw", results=())
    graph = _graph(
        retriever=FakeRetriever([misdetected]),
        grader=FakeGrader(False),
        settings=Settings(environment="test", agent_max_repairs=0),
    )

    result = asyncio.run(
        graph.answer("q", user_id="user-1", preferred_language="fr")
    )

    assert result.answer.language == "fr"


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


# --- the grounding gate ------------------------------------------------------------------------


class FakeGroundingJudge:
    """Scripted grounding verdicts; records exactly what it was asked to judge."""

    def __init__(self, supported: bool = True, *, error: Exception | None = None) -> None:
        self._supported = supported
        self._error = error
        self.calls: list[tuple[str, str]] = []  # (answer, context) per judgement

    def is_supported(self, *, answer: str, context: str) -> bool:
        self.calls.append((answer, context))
        if self._error is not None:
            raise self._error
        return self._supported


class TwoAnswerClient(FakeStreamClient):
    """Answers the grounded prompt and the refusal prompt differently.

    Discriminates on the real ``NO_CONTEXT_SYSTEM`` constant, so a test that expects a refusal
    cannot pass by accident just because both paths emit the same scripted text.
    """

    def __init__(self, grounded: str, refusal: str = "Your documents don't cover that.") -> None:
        super().__init__([grounded])
        self._refusal = refusal

    async def astream_completion(
        self, *, model: str, system: str, prompt: str, history: Sequence[ConversationTurn] = ()
    ) -> AsyncIterator[str]:
        self.systems.append(system)
        self.stream_prompts.append(prompt)
        yield self._refusal if system == NO_CONTEXT_SYSTEM else self._deltas[0]


def _gated(
    judge: FakeGroundingJudge,
    *,
    client: FakeStreamClient | None = None,
    retriever: FakeRetriever | None = None,
):
    return build_rag_graph(
        Settings(environment="test", grounding_gate=True),
        retriever=retriever or FakeRetriever(),
        client=client or TwoAnswerClient("Bharat ka rajdhaan Dilli hai. [1]"),
        grader=FakeGrader(True),
        judge=judge,
    )


def test_the_grounding_gate_replaces_an_unsupported_answer_with_a_refusal() -> None:
    # The manual-testing defect, reproduced: a health document is retrieved, graded relevant
    # (0.425 cosine clears a 0.0 floor), and the model answers "Dilli" from parametric knowledge
    # with [1] pointing at the health passage. A fabricated citation looks like evidence, which
    # is worse than a plainly wrong answer. Measured hallucination rate on unanswerable
    # questions: 61% (scripts/eval_refusal.py).
    judge = FakeGroundingJudge(supported=False)
    graph = _gated(judge)

    result = asyncio.run(graph.answer("bharat ka rajdhaan kya hai", user_id="user-1"))

    assert result.answer.answer == "Your documents don't cover that."
    assert result.answer.citations == ()


def test_the_grounding_gate_never_streams_the_text_of_an_unsupported_answer() -> None:
    # You cannot un-send a hallucination. A gate that fires only *after* the draft has streamed
    # token-by-token is decorative: the user has already read "Dilli", and the refusal replacing
    # it arrives too late to matter. So gated generation buffers — the whole reason the gate
    # costs latency and not just a provider call.
    graph = _gated(FakeGroundingJudge(supported=False))

    events = asyncio.run(_collect(graph.stream("bharat ka rajdhaan kya hai", user_id="user-1")))

    streamed = "".join(e.text for e in events if isinstance(e, Token))
    assert "Dilli" not in streamed
    assert streamed == "Your documents don't cover that."


def test_the_grounding_gate_delivers_a_supported_answer_with_its_citations() -> None:
    # The other half: buffering must not swallow the answers that pass. A gate that refuses
    # everything would score 0% hallucination and be useless.
    graph = _gated(FakeGroundingJudge(supported=True), client=TwoAnswerClient("Bharat is [1]."))

    events = asyncio.run(_collect(graph.stream("what is bharat", user_id="user-1")))

    done = [e for e in events if isinstance(e, Done)]
    assert "".join(e.text for e in events if isinstance(e, Token)) == "Bharat is [1]."
    assert len(done) == 1  # exactly one terminal event, whichever branch ran
    assert done[0].answer.citations[0].chunk_id == "doc-1:0"


def test_the_grounding_gate_fails_open_when_the_judge_itself_errors() -> None:
    # Fails open, like LlmRelevanceGrader. A rate limit or a 502 on the safety check is not
    # evidence the answer was wrong, and failing closed would convert a provider blip into every
    # answer in the product becoming "your documents don't cover this".
    judge = FakeGroundingJudge(
        error=AppError("judge down", code="faithfulness_judge_error", status_code=502)
    )
    graph = _gated(judge, client=TwoAnswerClient("Bharat is [1]."))

    result = asyncio.run(graph.answer("what is bharat", user_id="user-1"))

    assert result.answer.answer == "Bharat is [1]."
    assert result.answer.citations[0].chunk_id == "doc-1:0"


def test_the_longest_gated_path_stays_inside_the_recursion_limit() -> None:
    # The worst case got one super-step longer when ground_check was added: entry, condense,
    # route, retrieve, grade, repair, retrieve, grade, generate, ground_check,
    # generate_no_context. If the slack no longer covers it, the graph raises
    # agent_recursion_limit *instead of answering* — and only on the rare gated-repair-rejected
    # path, so nothing else in this suite would notice.
    graph = build_rag_graph(
        Settings(environment="test", grounding_gate=True, agent_max_repairs=1),
        retriever=FakeRetriever([_context(results=()), _context()]),
        client=TwoAnswerClient("Bharat ka rajdhaan Dilli hai. [1]"),
        grader=FakeGrader(False, True),  # one repair, then relevant
        judge=FakeGroundingJudge(supported=False),  # ...and then rejected
    )

    result = asyncio.run(
        graph.answer(
            "bharat ka rajdhaan kya hai",
            user_id="user-1",
            history=(ConversationTurn(role="user", content="earlier turn"),),  # forces condense
        )
    )

    assert result.answer.answer == "Your documents don't cover that."


def test_no_grounding_judge_is_built_by_default() -> None:
    # None, not a judge that always approves: the disabled gate must cost zero provider calls,
    # so ground_check short-circuits on the None rather than paying to be told yes.
    assert build_grounding_judge(Settings(environment="test")) is None


def test_the_grounding_gate_builds_the_faithfulness_judge_when_enabled() -> None:
    judge = build_grounding_judge(
        Settings(environment="test", grounding_gate=True, generation_api_key="k")
    )

    assert isinstance(judge, LlmFaithfulnessJudge)
