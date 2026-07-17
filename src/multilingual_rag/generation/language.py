"""Resolve which language an answer should be written in.

Guards against the "answer in unknown" bug: ``LanguageDetector`` returns ``"unknown"`` for
short queries (under 20 chars — most real questions), and the generator used to pass that
straight into the prompt. When the query language is unknown, fall back to the language of the
retrieved evidence, then English.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from multilingual_rag.core.models import VectorSearchResult

UNKNOWN_LANGUAGE = "unknown"
DEFAULT_LANGUAGE = "en"


def resolve_answer_language(
    preferred_language: str | None,
    query_language: str,
    results: Sequence[VectorSearchResult],
) -> str:
    """Pick the answer language: caller preference, else the query's, else the evidence's."""
    if preferred_language:
        return preferred_language
    if query_language and query_language != UNKNOWN_LANGUAGE:
        return query_language

    known = [
        result.language
        for result in results
        if result.language and result.language != UNKNOWN_LANGUAGE
    ]
    if known:
        return Counter(known).most_common(1)[0][0]
    return DEFAULT_LANGUAGE
