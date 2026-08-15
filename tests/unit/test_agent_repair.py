"""choose_repair table tests — a pure function, so no graph, no LLM, no fakes."""

from __future__ import annotations

from multilingual_rag.agent.repair import RepairStrategy, choose_repair, next_language
from multilingual_rag.retrieval.routing import LanguageRoute

HI_ONLY = ("hi",)
ALL_THREE = ("hi", "kn", "te")

ROMANIZED = "bharat ki rajdhani kya hai"
NATIVE = "भारत की राजधानी क्या है"


def _routed(target: str = "hi") -> LanguageRoute:
    """A route where transliteration actually happened."""
    return LanguageRoute(
        search_text=NATIVE, target_language=target, transliterated_query=NATIVE
    )


def _untouched(query: str = ROMANIZED) -> LanguageRoute:
    """A route that left the query alone."""
    return LanguageRoute(search_text=query, target_language=None, transliterated_query=None)


def _choose(
    *,
    route: LanguageRoute | None,
    question: str = ROMANIZED,
    languages: tuple[str, ...] = HI_ONLY,
    tried: tuple[RepairStrategy, ...] = (),
) -> RepairStrategy | None:
    return choose_repair(
        route=route, question=question, configured_languages=languages, tried=tried
    )


def test_a_transliterated_query_falls_back_to_the_raw_form_first() -> None:
    # The transliterated search is the one that just failed, and retrying raw is free —
    # no LLM call. eval_romanized.py measures a real population where transliterating hurt.
    assert _choose(route=_routed()) == "raw_fallback"


def test_with_one_configured_language_the_second_try_is_a_rewrite_not_a_relanguage() -> None:
    # Default config is hi only, so there is no other script to try — relanguage must not fire.
    assert _choose(route=_routed(), tried=("raw_fallback",)) == "rewrite"


def test_with_three_configured_languages_the_second_try_re_routes() -> None:
    assert (
        _choose(route=_routed("hi"), languages=ALL_THREE, tried=("raw_fallback",))
        == "relanguage"
    )


def test_a_native_script_query_is_never_raw_fallback_or_relanguage() -> None:
    # Nothing was transliterated and the text is not Latin — only a rewrite makes sense.
    assert _choose(route=_untouched(NATIVE), question=NATIVE, languages=ALL_THREE) == "rewrite"


def test_an_english_query_that_was_not_transliterated_goes_straight_to_rewrite() -> None:
    assert _choose(route=_untouched("what is the capital"), question="what is the capital") == (
        "rewrite"
    )


def test_every_strategy_exhausted_returns_none() -> None:
    assert (
        _choose(
            route=_routed("hi"),
            languages=ALL_THREE,
            tried=("raw_fallback", "relanguage", "rewrite"),
        )
        is None
    )


def test_a_strategy_is_never_repeated() -> None:
    # raw_fallback is still "available" by route shape, but it has been used.
    assert _choose(route=_routed(), tried=("raw_fallback", "rewrite")) is None


def test_no_route_yet_still_allows_a_rewrite() -> None:
    assert _choose(route=None) == "rewrite"


def test_next_language_skips_the_current_one() -> None:
    assert next_language(ALL_THREE, "hi") == "kn"
    assert next_language(HI_ONLY, "hi") is None
    assert next_language(HI_ONLY, None) == "hi"
