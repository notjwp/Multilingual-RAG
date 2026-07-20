"""Rewrite a follow-up into a standalone query for history-aware retrieval.

A conversational follow-up ("who founded it?") embeds poorly because the referent lives in earlier
turns. Before retrieval, a small "condense" LLM call rewrites it into a self-contained question so
the vector search hits the right chunks. The instruction preserves the original language/script so
the romanized-Indic detection + transliteration pipeline still fires on the rewritten query.
"""

from __future__ import annotations

from collections.abc import Sequence

from multilingual_rag.core.models import ConversationTurn

CONTEXTUALIZE_SYSTEM = (
    "You rewrite a follow-up question into a standalone question using the conversation. "
    "Resolve pronouns and implicit references so the question is fully self-contained. "
    "Do NOT answer it. Preserve the original language and script exactly — do not translate or "
    "transliterate. If the question is already standalone, return it unchanged. "
    "Return only the rewritten question, with no preamble or quotes."
)

_MAX_QUERY_CHARS = 400


def build_contextualize_prompt(history: Sequence[ConversationTurn], question: str) -> str:
    """Format recent turns + the follow-up into the condense prompt."""
    conversation = "\n".join(f"{turn.role.capitalize()}: {turn.content}" for turn in history)
    return (
        f"Conversation so far:\n{conversation}\n\n"
        f"Follow-up question:\n{question}\n\n"
        "Standalone question:"
    )


def clean_standalone_query(raw: str, *, fallback: str) -> str:
    """Normalize the model's rewrite; fall back to the original question if it's empty."""
    text = raw.strip().strip('"').strip("'").strip()
    if not text:
        return fallback
    return text[:_MAX_QUERY_CHARS]
