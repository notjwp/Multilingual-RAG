from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from openai import OpenAIError
from pydantic import SecretStr

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.embeddings.openai_embeddings import OpenAIEmbeddingProvider, validate_texts


@dataclass(frozen=True)
class FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


class FakeOpenAIClient:
    def __init__(self, *, fail: bool = False, missing_item: bool = False) -> None:
        self.fail = fail
        self.missing_item = missing_item
        self.requests: list[tuple[str, ...]] = []

    def create_embeddings(self, *, texts: Sequence[str], model: str) -> FakeEmbeddingResponse:
        del model
        if self.fail:
            raise OpenAIError("Connection failed")

        batch = tuple(texts)
        self.requests.append(batch)
        items = [
            FakeEmbeddingItem(index=index, embedding=[float(index), float(len(text))])
            for index, text in enumerate(batch)
        ]
        if self.missing_item:
            items = items[:-1]
        return FakeEmbeddingResponse(data=list(reversed(items)))


def test_embed_documents_batches_requests_and_preserves_order() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(
        Settings(openai_api_key=SecretStr("test-key"), openai_embedding_batch_size=2),
        client=client,
    )

    vectors = provider.embed_documents(("alpha", "bravo", "charlie"))

    assert client.requests == [("alpha", "bravo"), ("charlie",)]
    assert vectors == [[0.0, 5.0], [1.0, 5.0], [0.0, 7.0]]


def test_embed_query_embeds_single_text() -> None:
    provider = OpenAIEmbeddingProvider(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeOpenAIClient(),
    )

    assert provider.embed_query(" multilingual query ") == [0.0, 18.0]


def test_provider_requires_api_key_without_injected_client() -> None:
    with pytest.raises(AppError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(Settings(openai_api_key=None))


def test_validate_texts_rejects_empty_input() -> None:
    with pytest.raises(AppError, match="At least one text"):
        validate_texts((), allow_multiple=True)


def test_validate_texts_rejects_blank_text() -> None:
    with pytest.raises(AppError, match="must not contain blank"):
        validate_texts(("valid", "   "), allow_multiple=True)


def test_openai_errors_are_wrapped() -> None:
    provider = OpenAIEmbeddingProvider(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeOpenAIClient(fail=True),
    )

    with pytest.raises(AppError, match="OpenAI embedding request failed"):
        provider.embed_query("hello")


def test_mismatched_response_size_is_rejected() -> None:
    provider = OpenAIEmbeddingProvider(
        Settings(openai_api_key=SecretStr("test-key")),
        client=FakeOpenAIClient(missing_item=True),
    )

    with pytest.raises(AppError, match="did not match request size"):
        provider.embed_documents(("alpha", "bravo"))
