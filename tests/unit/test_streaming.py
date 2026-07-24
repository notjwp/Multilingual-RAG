"""StreamingAnswerGenerator unit test — fake stream client + fake retrieval (no network/model)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import ConversationTurn, RetrievalContext, VectorSearchResult
from multilingual_rag.generation.streaming import Done, StreamEvent, StreamingAnswerGenerator, Token


class FakeStreamClient:
    def __init__(self, deltas: Sequence[str], *, rewritten: str = "REWRITTEN") -> None:
        self._deltas = deltas
        self._rewritten = rewritten
        self.acomplete_calls = 0
        self.stream_history: tuple[object, ...] = ()

    async def astream_completion(
        self, *, model: str, system: str, prompt: str, history: object = ()
    ) -> AsyncIterator[str]:
        self.stream_history = tuple(history)  # type: ignore[arg-type]
        for delta in self._deltas:
            yield delta

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        self.acomplete_calls += 1
        return self._rewritten


class FakeRetrieval:
    def __init__(self, context: RetrievalContext) -> None:
        self._context = context
        self.queries: list[str] = []
        self.session_ids: list[str | None] = []

    def retrieve(
        self, query: str, *, user_id: str, session_id: str | None = None
    ) -> RetrievalContext:
        self.queries.append(query)
        self.session_ids.append(session_id)
        return self._context


def _context() -> RetrievalContext:
    result = VectorSearchResult(
        chunk_id="doc-1:0", document_id="doc-1", text="Bharat is India.", language="en",
        source="s.txt", chunk_index=0, score=0.9, page=1, token_count=4, metadata={},
    )
    return RetrievalContext(query="what is bharat", query_language="en", results=(result,))


async def _collect(generator: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in generator]


def test_stream_yields_tokens_then_grounded_done() -> None:
    generator = StreamingAnswerGenerator(
        Settings(environment="test"),
        retrieval_service=FakeRetrieval(_context()),  # type: ignore[arg-type]
        client=FakeStreamClient(["Bharat", " is ", "India [1]."]),
    )

    events = asyncio.run(_collect(generator.stream("what is bharat", user_id="user-1")))

    tokens = [event for event in events if isinstance(event, Token)]
    done = [event for event in events if isinstance(event, Done)]
    assert [token.text for token in tokens] == ["Bharat", " is ", "India [1]."]
    assert len(done) == 1
    answer = done[0].answer
    assert answer.answer == "Bharat is India [1]."
    assert answer.language == "en"
    assert answer.citations[0].chunk_id == "doc-1:0"  # cited [1] -> results[0]


def test_first_turn_skips_condense_and_searches_raw_query() -> None:
    retrieval = FakeRetrieval(_context())
    client = FakeStreamClient(["Bharat [1]."])
    generator = StreamingAnswerGenerator(
        Settings(environment="test"),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        client=client,
    )

    asyncio.run(_collect(generator.stream("what is bharat", user_id="user-1")))

    assert client.acomplete_calls == 0  # no condense LLM call with empty history
    assert retrieval.queries == ["what is bharat"]  # raw query searched
    assert client.stream_history == ()


def test_follow_up_condenses_then_searches_rewritten_query_with_history() -> None:
    retrieval = FakeRetrieval(_context())
    client = FakeStreamClient(["Wexler [1]."], rewritten="Who founded the Zorblax Protocol?")
    generator = StreamingAnswerGenerator(
        Settings(environment="test"),
        retrieval_service=retrieval,  # type: ignore[arg-type]
        client=client,
    )
    history = (
        ConversationTurn(role="user", content="Tell me about the Zorblax Protocol"),
        ConversationTurn(role="assistant", content="It is a fictional standard."),
    )

    events = asyncio.run(
        _collect(generator.stream("who founded it?", user_id="user-1", history=history))
    )

    assert client.acomplete_calls == 1  # one condense call
    assert retrieval.queries == ["Who founded the Zorblax Protocol?"]  # rewritten query searched
    assert client.stream_history == history  # generation received the prior turns
    done = [event for event in events if isinstance(event, Done)]
    assert done[0].answer.answer == "Wexler [1]."
