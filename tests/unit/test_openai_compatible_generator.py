"""Generation against an OpenAI-compatible endpoint: grounding + the error contract. No network."""

import asyncio
import json

import httpx
import pytest
from openai import NotFoundError, OpenAIError, RateLimitError
from pydantic import SecretStr

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import ConversationTurn, RetrievalContext, VectorSearchResult
from multilingual_rag.generation.openai_compatible_generator import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleChatClient,
    OpenAICompatibleStreamClient,
)


def _results(*chunk_ids: str) -> tuple[VectorSearchResult, ...]:
    return tuple(
        VectorSearchResult(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=f"text {chunk_id}",
            language="zh-cn",
            source="s.txt",
            chunk_index=index,
            score=0.9,
            token_count=2,
        )
        for index, chunk_id in enumerate(chunk_ids)
    )


def _context(query_language: str = "en") -> RetrievalContext:
    return RetrievalContext(query="q", query_language=query_language, results=_results("a", "b"))


def _api_error(kind: type, message: str) -> OpenAIError:
    """Build a real OpenAI SDK error (they require a response/body)."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status_code=429 if kind is RateLimitError else 404, request=request)
    return kind(message=message, response=response, body=None)


class FakeChatClient:
    def __init__(self, *, text: str = "Answer [1]", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, str, str]] = []
        self.histories: list[tuple[object, ...]] = []

    def create_completion(
        self, *, model: str, system: str, prompt: str, history: object = ()
    ) -> str:
        self.calls.append((model, system, prompt))
        self.histories.append(tuple(history))  # type: ignore[arg-type]
        if self.error is not None:
            raise self.error
        return self.text


def _generator(client: FakeChatClient) -> OpenAICompatibleAnswerGenerator:
    return OpenAICompatibleAnswerGenerator(
        Settings(environment="test", generation_model="test/model"),
        client=client,
    )


def test_generates_answer_and_cites_only_marked_chunks() -> None:
    generator = _generator(FakeChatClient(text="Grounded claim [2]."))

    answer = generator.generate_answer(context=_context(), preferred_language="es")

    assert answer.answer == "Grounded claim [2]."
    assert answer.language == "es"
    assert [c.chunk_id for c in answer.citations] == ["b"]  # only [2], not everything


def test_uses_configured_model_and_system_prompt() -> None:
    client = FakeChatClient()
    _generator(client).generate_answer(context=_context())

    model, system, prompt = client.calls[0]
    assert model == "test/model"
    assert "retrieval-augmented" in system.lower()
    assert "Answer language: en" in prompt


def test_never_answers_in_unknown() -> None:
    # Short query -> "unknown"; must fall back to the evidence's language.
    generator = _generator(FakeChatClient())
    answer = generator.generate_answer(context=_context(query_language="unknown"))
    assert answer.language == "zh-cn"


def test_rate_limit_maps_to_429() -> None:
    generator = _generator(FakeChatClient(error=_api_error(RateLimitError, "slow down")))

    with pytest.raises(AppError) as exc:
        generator.generate_answer(context=_context())

    assert exc.value.code == "generation_rate_limited"
    assert exc.value.status_code == 429


def test_missing_model_names_the_env_var_to_fix() -> None:
    # Model catalogs rotate; a vanished model must be actionable, not a mystery.
    generator = _generator(FakeChatClient(error=_api_error(NotFoundError, "no such model")))

    with pytest.raises(AppError) as exc:
        generator.generate_answer(context=_context())

    assert exc.value.code == "generation_model_unavailable"
    assert "GENERATION_MODEL" in exc.value.message
    assert "test/model" in exc.value.message


def test_generic_errors_are_wrapped() -> None:
    generator = _generator(FakeChatClient(error=OpenAIError("boom")))

    with pytest.raises(AppError) as exc:
        generator.generate_answer(context=_context())

    assert exc.value.code == "generation_error"


def test_empty_answer_is_rejected() -> None:
    generator = _generator(FakeChatClient(text="   "))

    with pytest.raises(AppError, match="empty answer"):
        generator.generate_answer(context=_context())


def test_requires_key_without_injected_client() -> None:
    with pytest.raises(AppError, match="GENERATION_API_KEY"):
        OpenAICompatibleAnswerGenerator(Settings(environment="test", generation_api_key=None))


# Condensing moved to the agent graph's `condense` node — see tests/unit/test_agent_graph.py
# (`test_first_turn_skips_condense_...` and `test_follow_up_condenses_...`). This generator is now
# only the eval harness's blocking answer path.


def test_generate_answer_forwards_history_to_client() -> None:
    client = FakeChatClient(text="Answer [1]")
    history = (ConversationTurn(role="user", content="hi"),)

    _generator(client).generate_answer(context=_context(), history=history)

    assert client.histories[-1] == history


def test_provider_is_just_a_url_not_a_code_path() -> None:
    """The whole point of the refactor: GENERATION_BASE_URL selects the provider."""
    for base_url in (
        "https://integrate.api.nvidia.com/v1",  # NVIDIA NIM
        "https://openrouter.ai/api/v1",  # OpenRouter
        "http://localhost:11434/v1",  # local Ollama
    ):
        generator = OpenAICompatibleAnswerGenerator(
            Settings(
                environment="test",
                generation_api_key=SecretStr("k"),
                generation_base_url=base_url,
            )
        )
        client = generator.client
        assert isinstance(client, OpenAICompatibleChatClient)
        # The SDK normalises to a trailing slash; the configured host must be what we talk to.
        assert str(client._client.base_url).rstrip("/") == base_url.rstrip("/")


def test_a_malformed_stream_frame_becomes_an_apperror_not_a_valueerror() -> None:
    """Regression from a live run: the SDK raises json.JSONDecodeError (a ValueError) on a bad SSE
    frame. That is not an OpenAIError, so it slipped past every handler, escaped the graph, and
    reached chat_stream.py — which only renders `event: error` for AppError. The stream simply
    stopped, leaving the UI on a half-written bubble."""

    class _CorruptStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise json.JSONDecodeError("Expecting value", "", 0)

    class _Client:
        class chat:  # noqa: N801 - mirrors the SDK's attribute layout
            class completions:
                @staticmethod
                async def create(**_kwargs):
                    return _CorruptStream()

    client = OpenAICompatibleStreamClient.__new__(OpenAICompatibleStreamClient)
    client._client = _Client()  # type: ignore[attr-defined]

    async def _drain() -> None:
        async for _ in client.astream_completion(model="m", system="s", prompt="p"):
            pass

    with pytest.raises(AppError) as caught:
        asyncio.run(_drain())

    assert caught.value.code == "generation_stream_corrupt"
    assert caught.value.status_code == 502
