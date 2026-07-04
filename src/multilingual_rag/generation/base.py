"""Generation provider contracts."""

from __future__ import annotations

from typing import Protocol

from multilingual_rag.core.models import GeneratedAnswer, RetrievalContext


class AnswerGenerator(Protocol):
    """Protocol for grounded answer generators."""

    def generate_answer(
        self,
        *,
        context: RetrievalContext,
        preferred_language: str | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer grounded in retrieved context."""
        ...

