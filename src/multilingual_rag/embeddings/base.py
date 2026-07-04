"""Embedding provider contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

EmbeddingVector = list[float]


class EmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed document texts in provider-specific batches."""
        ...

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single query string."""
        ...

