"""Unit tests for the romanized-Indic transliteration layer (no network, no model load)."""

from __future__ import annotations

import pytest

from multilingual_rag.core.config import Settings
from multilingual_rag.transliteration.detect import is_romanized_indic
from multilingual_rag.transliteration.factory import build_transliterator
from multilingual_rag.transliteration.google import GoogleTransliterator
from multilingual_rag.transliteration.indicxlit import IndicXlitTransliterator, _clean
from multilingual_rag.transliteration.rule_based import SanscriptTransliterator
from multilingual_rag.transliteration.script import is_latin_script

_HI = ("hi",)


# --- is_latin_script -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bharat ki rajdhani kya hai", True),  # romanized Hindi -> attempt transliteration
        ("hello world", True),  # real English -> also dual-queried, but raw wins (harmless)
        ("भारत की राजधानी", False),  # already native Devanagari -> skip
        ("hello भारत", False),  # any Devanagari present -> skip
        ("中文查询", False),  # CJK -> not romanized Indic
        ("คำถาม", False),  # Thai -> skip
        ("12345 !!!", False),  # no letters at all -> skip
        ("", False),
    ],
)
def test_is_latin_script(text: str, expected: bool) -> None:
    assert is_latin_script(text) is expected


# --- is_romanized_indic (decides *whether* to transliterate) ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "bharat ki rajdhani kya hai",  # natural typing
        "panda kya khata hai",
        "dilli kahan hai",
        "bharata ki rajadhani kya hai",  # IAST-strip form (what the eval romanizer emits)
        "kauna sa desh hai",
    ],
)
def test_is_romanized_indic_detects_hindi(text: str) -> None:
    assert is_romanized_indic(text, _HI) is True


@pytest.mark.parametrize(
    "text",
    [
        "what is machine learning",  # plain English -> must NOT transliterate
        "the capital of france",  # "the" must not be a marker
        "who is the president",
        "list the largest cities in europe",
        "भारत की राजधानी क्या है",  # already native Devanagari -> not Latin
        "中文查询",  # CJK
    ],
)
def test_is_romanized_indic_rejects_non_hindi(text: str) -> None:
    assert is_romanized_indic(text, _HI) is False


def test_is_romanized_indic_off_when_hi_not_configured() -> None:
    assert is_romanized_indic("bharat kya hai", ()) is False
    assert is_romanized_indic("bharat kya hai", ("kn",)) is False


# --- rule-based adapter ----------------------------------------------------------------------


def test_rule_based_produces_devanagari_and_passes_through_unsupported() -> None:
    out = SanscriptTransliterator().transliterate("bharat", target_language="hi")
    assert out != "bharat"
    assert any("ऀ" <= ch <= "ॿ" for ch in out)  # contains Devanagari
    # Unsupported language / empty input are returned unchanged.
    assert SanscriptTransliterator().transliterate("bharat", target_language="fr") == "bharat"
    assert SanscriptTransliterator().transliterate("   ", target_language="hi") == "   "


# --- indicxlit output cleanup (no model load) ------------------------------------------------


def test_indicxlit_clean_trims_danda_and_duplicated_trailing_token() -> None:
    assert _clean("मेरा नाम क्या है है") == "मेरा नाम क्या है"  # dedup trailing repeat
    assert _clean("पांडा क्या खाता है।") == "पांडा क्या खाता है"  # strip danda


def test_indicxlit_passes_through_unsupported_language_without_loading() -> None:
    # target "fr" is unsupported -> returns input, never touching the model.
    assert IndicXlitTransliterator().transliterate("hello", target_language="fr") == "hello"


# --- google adapter fallback (no network) ----------------------------------------------------


def test_google_falls_back_to_rule_based_on_failure() -> None:
    translit = GoogleTransliterator(fallback=SanscriptTransliterator())

    async def _boom(text: str, dest: str) -> str:
        raise RuntimeError("network down")

    translit._translate = _boom  # type: ignore[method-assign]
    out = translit.transliterate("bharat", target_language="hi")

    assert out == SanscriptTransliterator().transliterate("bharat", target_language="hi")
    assert any("ऀ" <= ch <= "ॿ" for ch in out)  # fell back to real Devanagari


def test_google_passes_through_unsupported_language_without_network() -> None:
    assert GoogleTransliterator().transliterate("hello", target_language="fr") == "hello"


# --- factory selection -----------------------------------------------------------------------


def test_factory_returns_none_when_disabled_or_off() -> None:
    assert build_transliterator(Settings(transliteration_enabled=False)) is None
    assert build_transliterator(Settings(transliteration_provider="off")) is None


def test_factory_selects_provider() -> None:
    from multilingual_rag.transliteration.google import GoogleTransliterator as G

    assert isinstance(build_transliterator(Settings()), G)  # default = google
    assert isinstance(
        build_transliterator(Settings(transliteration_provider="rule-based")),
        SanscriptTransliterator,
    )
    assert isinstance(
        build_transliterator(Settings(transliteration_provider="indicxlit")),
        IndicXlitTransliterator,
    )


def test_factory_llm_without_key_degrades_to_rule_based() -> None:
    # No GENERATION_API_KEY -> can't reach the LLM, so fall back to the guaranteed-local adapter.
    built = build_transliterator(Settings(transliteration_provider="llm", generation_api_key=None))
    assert isinstance(built, SanscriptTransliterator)
