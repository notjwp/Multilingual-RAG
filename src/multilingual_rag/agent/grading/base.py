"""Relevance grading contract (a port, like ``EmbeddingProvider`` and ``VectorStore``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from multilingual_rag.core.models import VectorSearchResult


@dataclass(frozen=True)
class Grade:
    """The verdict on one retrieval attempt.

    ``reason`` is short and human-readable because it becomes the agent step's ``detail`` in the
    chat UI — it is shown to a user, not only logged.
    """

    relevant: bool
    reason: str
    top_score: float | None


class RelevanceGrader(Protocol):
    """Decide whether retrieved chunks are good enough to answer from.

    Async because the grader runs inside the agent graph and the LLM adapter reuses the
    ``StreamClient`` the graph already holds — no second client, no extra API-key branch, and no
    thread hop. The free adapter is ``async def`` with no awaits, which is fine.
    """

    async def grade(
        self, *, query: str, results: Sequence[VectorSearchResult]
    ) -> Grade:
        """Return whether ``results`` plausibly answer ``query``."""
        ...
