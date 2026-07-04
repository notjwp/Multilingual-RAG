"""OpenAI embedding provider implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from fastapi import status
from openai import OpenAI, OpenAIError

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.embeddings.base import EmbeddingVector


class _EmbeddingItem(Protocol):
    @property
    def index(self) -> int:
        """Return the item's input index."""
        ...

    @property
    def embedding(self) -> Sequence[float]:
        """Return the embedding vector."""
        ...


class _EmbeddingResponse(Protocol):
    @property
    def data(self) -> Sequence[_EmbeddingItem]:
        """Return embedding items."""
        ...


class _OpenAIEmbeddingClient(Protocol):
    def create_embeddings(self, *, texts: Sequence[str], model: str) -> _EmbeddingResponse:
        """Create embeddings for one batch of text."""
        ...


class OpenAIEmbeddingClient:
    """Small adapter around the OpenAI SDK embedding resource."""

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def create_embeddings(self, *, texts: Sequence[str], model: str) -> _EmbeddingResponse:
        """Create embeddings using the configured OpenAI client."""
        response = self._client.embeddings.create(input=list(texts), model=model)
        return cast(_EmbeddingResponse, response)


class OpenAIEmbeddingProvider:
    """Embed text using OpenAI embeddings."""

    def __init__(self, settings: Settings, *, client: _OpenAIEmbeddingClient | None = None) -> None:
        api_key = settings.openai_api_key
        if client is None and api_key is None:
            raise AppError(
                "OPENAI_API_KEY is required to create OpenAI embeddings.",
                code="missing_openai_api_key",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self.model = settings.openai_embedding_model
        self.batch_size = settings.openai_embedding_batch_size
        if client is not None:
            self.client = client
        else:
            if api_key is None:
                raise AssertionError("api_key should be set after validation")
            self.client = OpenAIEmbeddingClient(api_key.get_secret_value())

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed document texts in batches while preserving input order."""
        normalized_texts = validate_texts(texts, allow_multiple=True)
        vectors: list[EmbeddingVector] = []

        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return vectors

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query string."""
        return self._embed_batch(validate_texts((text,), allow_multiple=False))[0]

    def _embed_batch(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        try:
            response = self.client.create_embeddings(texts=texts, model=self.model)
        except OpenAIError as exc:
            raise AppError(
                "OpenAI embedding request failed.",
                code="openai_embedding_error",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

        ordered_items = sorted(response.data, key=lambda item: item.index)
        if len(ordered_items) != len(texts):
            raise AppError(
                "OpenAI embedding response did not match request size.",
                code="invalid_embedding_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        vectors = [list(item.embedding) for item in ordered_items]
        if any(not vector for vector in vectors):
            raise AppError(
                "OpenAI embedding response included an empty vector.",
                code="invalid_embedding_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        return vectors


def validate_texts(texts: Sequence[str], *, allow_multiple: bool) -> tuple[str, ...]:
    """Validate and normalize text inputs before embedding."""
    if not texts:
        raise AppError(
            "At least one text value is required for embedding.",
            code="empty_embedding_input",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    normalized_texts = tuple(text.strip() for text in texts)
    if any(not text for text in normalized_texts):
        raise AppError(
            "Embedding inputs must not contain blank text.",
            code="blank_embedding_input",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not allow_multiple and len(normalized_texts) != 1:
        raise AppError(
            "Query embedding requires exactly one text value.",
            code="invalid_query_embedding_input",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return normalized_texts
