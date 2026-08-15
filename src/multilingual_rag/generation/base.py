"""Generation provider contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
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


class StreamClient(Protocol):
    """A chat client that streams the assistant's reply as text deltas.

    The agent graph depends on this for both the answer stream and its one-shot calls (condense,
    query rewrite, and the opt-in LLM relevance grader), so a single fake covers all of them.
    """

    def astream_completion(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        """Yield assistant message deltas for a system + history + user exchange."""
        ...

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        """Return a whole (non-streamed) completion — the condense/rewrite/grade call."""
        ...

