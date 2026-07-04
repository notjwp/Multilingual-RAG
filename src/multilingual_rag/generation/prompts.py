"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.retrieval.context import format_context

SYSTEM_INSTRUCTIONS = (
    "You are a multilingual retrieval-augmented generation assistant. "
    "Answer only from the provided context. If the context is insufficient, say so. "
    "Preserve factual details and cite supporting chunks by their bracket numbers."
)


def build_answer_prompt(context: RetrievalContext, *, response_language: str) -> str:
    """Build the user prompt for answer generation."""
    formatted_context = format_context(context)
    return (
        f"Answer language: {response_language}\n\n"
        f"Question:\n{context.query}\n\n"
        f"Retrieved context:\n{formatted_context or 'No context was retrieved.'}\n\n"
        "Return a concise answer followed by citations where relevant."
    )

