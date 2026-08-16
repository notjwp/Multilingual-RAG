"""Synthesize the romanized queries the Indic retrieval eval scores against.

No romanized XQuAD exists, so `scripts/eval_romanized.py` has to manufacture "what a person would
type" from the native-script questions. How that is done decides whether the evaluation is honest.

**The flaw this module exists to fix.** It used to be done entirely with
``indic_transliteration.sanscript`` (Devanagari -> IAST -> strip diacritics). But the ``rule-based``
transliteration adapter *under test* is that same library, so it was being scored on inverting the
exact character-level transform that produced its input. Measured: rule-based scored 0.950 against
google's 0.700 on identical queries; after switching to human romanizations the gap fell to
0.850 vs 0.800. A quarter of that lead was the harness marking its own homework.

The old scheme was also simply unlike real input. Romanized Hindi keeps English loanwords in
English, and spells function words phonetically rather than scholastically:

    native     जोश नॉर्मन ने कितने बॉल को इंटरसेप्ट किया?
    sanscript  josa narmana ne kitane bala ko imtarasepta kiya?
    human      josh norman ne kitane ball co intercept kiya?

So :func:`romanize` now substitutes human-attested spellings word by word, from a committed
lexicon built by ``scripts/build_romanization_lexicon.py`` out of Dakshina plus a word-aligned
parallel corpus — neither of which involves ``indic_transliteration``. Vocabulary the lexicon
lacks still falls back to :func:`rule_romanize`, which leaves a bounded residual advantage for the
rule-based adapter; callers pass ``stats`` and report that share rather than hiding it.

This lives in the package rather than in ``scripts/`` so it is covered by the test suite and by
``mypy --strict`` — it is load-bearing evaluation logic, not glue.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from indic_transliteration import sanscript  # type: ignore[import-untyped]

ROMANIZE_SCHEME: dict[str, str] = {
    "hi": sanscript.DEVANAGARI,
    "kn": sanscript.KANNADA,
    "te": sanscript.TELUGU,
}

# Devanagari, Kannada and Telugu blocks. Digits are inside these ranges deliberately: the lexicon
# maps १ -> 1, which is what people type.
NATIVE_WORD = re.compile(r"[ऀ-ॿಀ-೿ఀ-౿]+")

# repo-root/data/eval/... — this file is src/multilingual_rag/evaluation/romanization.py.
LEXICON_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"


@lru_cache(maxsize=4)
def human_lexicon(lang: str) -> Mapping[str, str]:
    """Human-written romanizations for ``lang``: native word -> spelling.

    Empty when no lexicon has been built for that language (only ``hi`` ships one today), which
    makes :func:`romanize` degrade to pure ``rule_romanize`` rather than fail.
    """
    path = LEXICON_DIR / f"romanization_{lang}.json"
    if not path.exists():
        return {}
    loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def rule_romanize(text: str, lang: str = "hi") -> str:
    """Native Indic script -> diacritic-stripped ASCII, via ``indic_transliteration``.

    **Never use this alone as the evaluation's query source** — see the module docstring. It is
    the inverse of the ``rule-based`` adapter's own scheme, so scoring that adapter against it
    measures a round trip through one library rather than transliteration quality. Kept as the
    per-word fallback for vocabulary the human lexicon lacks.
    """
    iast = sanscript.transliterate(text, ROMANIZE_SCHEME[lang], sanscript.IAST)
    decomposed = unicodedata.normalize("NFKD", iast)
    # Keep ASCII only: drop combining diacritics AND any native matra sanscript left untranslated.
    ascii_only = "".join(
        ch for ch in decomposed if ord(ch) < 128 and not unicodedata.combining(ch)
    )
    return ascii_only.lower()


def romanize(text: str, lang: str = "hi", *, stats: Counter[str] | None = None) -> str:
    """Native Indic script -> the romanization a person would actually type.

    Word by word: a human-attested spelling when the lexicon has one, else
    :func:`rule_romanize`. Non-native characters (Latin, ASCII digits, punctuation) pass through.

    ``stats`` accumulates ``hit``/``oov`` counts so a run can report how much of its query text
    still came from the rule-based fallback — the residual bias toward that adapter.
    """
    lexicon = human_lexicon(lang)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        spelling = lexicon.get(word)
        if stats is not None:
            stats["hit" if spelling else "oov"] += 1
        return spelling if spelling else rule_romanize(word, lang)

    return NATIVE_WORD.sub(replace, text).lower()
