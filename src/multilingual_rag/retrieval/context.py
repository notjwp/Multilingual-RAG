"""Context formatting for grounded answer generation."""

from __future__ import annotations

from multilingual_rag.core.models import RetrievalContext


def format_context(context: RetrievalContext) -> str:
    """Format retrieved chunks into a compact cited context block."""
    lines: list[str] = []
    for index, result in enumerate(context.results, start=1):
        page = f", page {result.page}" if result.page is not None else ""
        lines.append(
            f"[{index}] chunk_id={result.chunk_id}; source={result.source}{page}; "
            f"language={result.language}; score={result.score:.4f}\n{result.text}"
        )
    return "\n\n".join(lines)

