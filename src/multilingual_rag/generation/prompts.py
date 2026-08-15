"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from multilingual_rag.core.models import RetrievalContext
from multilingual_rag.retrieval.context import format_context

SYSTEM_INSTRUCTIONS = (
    "You are a multilingual retrieval-augmented generation assistant. "
    "Answer only from the provided context. If the context is insufficient, say so. "
    "Preserve factual details and cite supporting chunks by their bracket numbers."
)


NO_CONTEXT_SYSTEM = (
    "You are a multilingual retrieval-augmented assistant. A search of the user's own documents "
    "found nothing that answers their question, including after retrying. Tell them so plainly "
    "and briefly, in the requested language. Do NOT answer from your own knowledge, do not "
    "speculate about the answer, and do not cite anything."
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


def build_no_context_prompt(question: str, *, response_language: str) -> str:
    """Build the prompt for the give-up path, when retrieval stayed weak after repairs.

    This *sharpens* rather than adds behaviour — ``SYSTEM_INSTRUCTIONS`` already tells the model
    to say so when context is insufficient, and ``build_answer_prompt`` already emits "No context
    was retrieved." What the dedicated path guarantees is empty citations and a prompt with no
    retrieved text in it at all, so the answer cannot drift into parametric knowledge.
    """
    return (
        f"Answer language: {response_language}\n\n"
        f"Question the documents did not cover:\n{question}\n\n"
        "Tell the user their documents do not contain this, in one or two sentences."
    )

