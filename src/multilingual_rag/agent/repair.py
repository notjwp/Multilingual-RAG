"""Choosing *how* to retry after a weak retrieval.

This is the node that makes the graph specific to this project rather than a generic
grade-and-retry loop. When a retrieval comes back weak the first question worth asking here is
not "rewrite the query?" but **"was my script routing wrong?"** — the pipeline's whole job is
deciding whether a Latin-script query is really romanized Indic, and that decision can be wrong.

Kept a pure function so the interesting logic is a table test with no graph, no LLM, no fakes.
"""

from __future__ import annotations

from typing import Literal

from multilingual_rag.retrieval.routing import LanguageRoute
from multilingual_rag.transliteration.script import is_latin_script

RepairStrategy = Literal["raw_fallback", "relanguage", "rewrite"]

# Plain-language names for the agent step detail shown in the chat UI.
LANGUAGE_NAMES: dict[str, str] = {"hi": "Hindi", "kn": "Kannada", "te": "Telugu"}

REPAIR_REWRITE_SYSTEM = (
    "A document search found nothing relevant for the user's query. Rewrite it into a short, "
    "keyword-dense search query that is more likely to match document text. Keep the original "
    "language and script exactly — do not translate or transliterate. Do NOT answer the "
    "question. Return only the rewritten query, with no preamble or quotes."
)


def build_repair_rewrite_prompt(query: str) -> str:
    """Format the failed query into the repair-rewrite prompt."""
    return f"Query that found nothing:\n{query}\n\nRewritten search query:"


def language_name(code: str | None) -> str:
    """Return a human-readable language name for a code, for step details."""
    if code is None:
        return "the original language"
    return LANGUAGE_NAMES.get(code, code)


def next_language(configured_languages: tuple[str, ...], current: str | None) -> str | None:
    """Return another configured Indic language to try, or None when there isn't one."""
    for language in configured_languages:
        if language != current:
            return language
    return None


def choose_repair(
    *,
    route: LanguageRoute | None,
    question: str,
    configured_languages: tuple[str, ...],
    tried: tuple[RepairStrategy, ...],
) -> RepairStrategy | None:
    """Pick the next repair strategy, cheapest-and-most-evidenced first, or None when exhausted.

    1. ``raw_fallback`` — the query *was* transliterated, so the transliterated search is the
       thing that just failed. Retry the raw form. Free: one embed, one search, no LLM call.
       This is not decorative — ``scripts/eval_romanized.py`` measures shipped retention at 0.747
       against native 1.0, so there is a measured population of queries where transliterating
       made retrieval worse.
    2. ``relanguage`` — a Latin-script query with more than one configured Indic language: the
       detector may have picked the wrong one. Re-route to another and transliterate to that
       script. With the default ``TRANSLITERATION_LANGUAGES=("hi",)`` this correctly never fires;
       it comes alive with ``hi,kn,te`` plus the muril or google detector.
    3. ``rewrite`` — always available. One LLM call, a *different* prompt from condense.

    A strategy is never tried twice in one turn.
    """
    if "raw_fallback" not in tried and route is not None and route.transliteration_applied:
        return "raw_fallback"

    if (
        "relanguage" not in tried
        and is_latin_script(question)
        and next_language(configured_languages, route.target_language if route else None)
        is not None
        and len(configured_languages) > 1
    ):
        return "relanguage"

    if "rewrite" not in tried:
        return "rewrite"

    return None
