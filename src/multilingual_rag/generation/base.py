"""Generation provider contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from multilingual_rag.core.models import ConversationTurn, GeneratedAnswer, RetrievalContext


class AnswerGenerator(Protocol):
    """Protocol for grounded answer generators."""

    def generate_answer(
        self,
        *,
        context: RetrievalContext,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> GeneratedAnswer:
        """Generate an answer grounded in retrieved context, optionally with prior turns."""
        ...

    def contextualize(self, history: Sequence[ConversationTurn], question: str) -> str:
        """Rewrite a follow-up into a standalone query using history (identity if no history)."""
        ...

