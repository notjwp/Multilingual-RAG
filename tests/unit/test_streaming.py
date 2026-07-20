"""StreamingAnswerGenerator unit test — fake stream client + fake retrieval (no network/model)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import RetrievalContext, VectorSearchResult
from multilingual_rag.generation.streaming import Done, StreamEvent, StreamingAnswerGenerator, Token


class FakeStreamClient:
    def __init__(self, deltas: Sequence[str]) -> None:
        self._deltas = deltas

    async def astream_completion(
        self, *, model: str, system: str, prompt: str
    ) -> AsyncIterator[str]:
        for delta in self._deltas:
            yield delta


class FakeRetrieval:
    def __init__(self, context: RetrievalContext) -> None:
        self._context = context

    def retrieve(self, query: str, *, user_id: str) -> RetrievalContext:
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
