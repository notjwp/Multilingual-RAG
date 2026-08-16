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

RepairStrategy = Literal["relanguage", "rewrite", "raw_fallback"]

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
    """Pick the next repair strategy, or None when they are exhausted.

    **The ordering was set by measurement, against my first instinct.** The original version led
    with ``raw_fallback`` on the story "was my script routing wrong?" — which sounded right and
    was wrong. The data says script routing is almost always *correct*: detection scores 98.3%
    and transliteration lifts recall 0.500 -> 0.917 on XQuAD-hi. So a failed romanized query is
    usually one that was correctly identified and then rendered imperfectly, not one that should
    have been left in Latin script. Leading with the raw form retreats to the worst option on the
    board, and end-to-end it lost: recall@5 0.767 against 0.800 for no agent at all.

    1. ``relanguage`` — a Latin-script query with more than one configured Indic language: the
       detector may have picked the wrong one. With the default hi-only
       ``TRANSLITERATION_LANGUAGES`` this correctly never fires; it comes alive with ``hi,kn,te``
       plus the muril or google detector.
    2. ``rewrite`` — one LLM call, a *different* prompt from condense.
    3. ``raw_fallback`` — last resort. There *is* a small population where transliteration hurt,
       so it is not worthless; it is just a bad first guess.

    A strategy is never tried twice in one turn.

    **Note on scope.** With the default grader (``score-threshold`` at a 0.0 floor) this only runs
    when retrieval returned *nothing at all*, so every strategy here is strictly safe — there is
    no incumbent result to damage. See ``agent/grading/score_threshold.py`` for why the floor is
    where it is.
    """
    transliterated = route is not None and route.transliteration_applied

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

    if "raw_fallback" not in tried and transliterated:
        return "raw_fallback"

    return None
