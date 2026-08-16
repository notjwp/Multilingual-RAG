"""Romanization of eval queries — the piece that decides whether the Indic eval is honest.

The regression these guard against: queries were once generated purely by
``indic_transliteration.sanscript``, which is also the ``rule-based`` adapter under test, so that
adapter was scored on inverting its own scheme (0.950 vs google's 0.700 on identical queries).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from multilingual_rag.evaluation.romanization import (
    LEXICON_DIR,
    human_lexicon,
    romanize,
    rule_romanize,
)

# Words the committed hi lexicon covers, with the spelling a person actually types.
BALL = "बॉल"
IN = "में"
TEAM = "टीम"


def test_the_committed_hindi_lexicon_loads() -> None:
    lexicon = human_lexicon("hi")

    assert len(lexicon) > 1000, "data/eval/romanization_hi.json missing or truncated"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in lexicon.items())


def test_a_language_without_a_lexicon_degrades_instead_of_failing() -> None:
    # Only hi ships one. kn/te must still romanize, via the rule-based fallback.
    assert human_lexicon("kn") == {}
    assert romanize("ಕನ್ನಡ", "kn")  # non-empty, no exception


def test_english_loanwords_are_spelled_in_english_not_phonetically() -> None:
    # The single most important property: real romanized Hindi writes "ball", never "bala".
    assert romanize(BALL) == "ball"
    assert rule_romanize(BALL) == "bala"


def test_function_words_use_the_spelling_people_type() -> None:
    assert romanize(IN) == "mein"
    assert rule_romanize(IN) == "mem"


def test_covered_words_do_not_reproduce_the_rule_based_scheme() -> None:
    """The bias check. For vocabulary we have human data on, output must differ from the
    rule-based adapter's own inverse — otherwise the eval is still marking its own homework."""
    differing = [w for w in (BALL, IN, TEAM) if romanize(w) != rule_romanize(w)]

    assert differing == [BALL, IN, TEAM]


def test_unknown_words_fall_back_to_the_rule_based_romanizer() -> None:
    # A nonsense Devanagari string cannot be in any human corpus.
    invented = "ज़्क्ष्वट"

    assert invented not in human_lexicon("hi")
    assert romanize(invented) == rule_romanize(invented)


def test_stats_count_human_hits_against_rule_based_fallbacks() -> None:
    stats: Counter[str] = Counter()

    romanize(f"{BALL} ज़्क्ष्वट {IN}", stats=stats)

    assert stats["hit"] == 2  # बॉल and में are covered
    assert stats["oov"] == 1  # the invented word is not


def test_stats_are_optional() -> None:
    assert romanize(BALL) == "ball"  # no stats argument, no error


def test_non_native_characters_pass_through() -> None:
    # ASCII digits, Latin and punctuation are not romanized — only native script is substituted.
    assert romanize(f"2015 {IN} NFL?") == "2015 mein nfl?"


def test_a_whole_question_mixes_human_and_fallback_spellings() -> None:
    stats: Counter[str] = Counter()

    result = romanize("जोश नॉर्मन ने कितने बॉल को इंटरसेप्ट किया?", stats=stats)

    # Proper nouns and loanwords survive as English, which is the entire point.
    assert "josh norman" in result
    assert "ball" in result
    assert "intercept" in result
    assert stats["hit"] >= 5


def test_output_is_lowercase_ascii() -> None:
    result = romanize("जोश नॉर्मन ने कितने बॉल को इंटरसेप्ट किया?")

    assert result == result.lower()
    assert result.isascii(), "a romanized query must contain no native script"


def test_rule_romanize_strips_diacritics_to_plain_ascii() -> None:
    # bhārata -> bharata: deliberately lossy, mirroring pure-ASCII keyboard input.
    assert rule_romanize("भारत") == "bharata"


def test_lexicon_is_cached_so_the_file_is_read_once() -> None:
    assert human_lexicon("hi") is human_lexicon("hi")


def test_the_committed_lexicon_is_well_formed_on_disk() -> None:
    path = LEXICON_DIR / "romanization_hi.json"
    assert path.exists(), "run scripts/build_romanization_lexicon.py"

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(loaded, dict)
    # Values are what gets typed: lowercase ASCII, no native script leaking back in.
    offenders = [v for v in loaded.values() if not v.isascii() or v != v.lower()]
    assert offenders == []


@pytest.mark.parametrize("lang", ["hi", "kn", "te"])
def test_every_supported_language_romanizes_without_error(lang: str) -> None:
    assert isinstance(romanize("परीक्षण", lang), str)


def test_lexicon_path_resolves_to_the_repo_data_directory() -> None:
    # The module is three levels inside src/, so a wrong parents[] index silently yields an empty
    # lexicon and a silently-rigged eval — worth pinning.
    assert Path(__file__).resolve().parents[2] / "data" / "eval" == LEXICON_DIR
