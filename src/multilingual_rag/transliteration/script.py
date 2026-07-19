"""Cheap script detection — the entire "detector" under dual-query."""

from __future__ import annotations

import re

# Non-Latin letter blocks that mean "not romanized Indic, skip transliteration": Devanagari
# (our target script — a query already in it is native), plus CJK, Hiragana/Katakana, Hangul,
# and Thai, so zh/ja/ko/th queries don't get needlessly dual-queried.
_NON_LATIN = re.compile(r"[ऀ-ॿ一-鿿぀-ヿ가-힯฀-๿]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def is_latin_script(text: str) -> bool:
    """True when ``text`` is Latin-script (has ASCII letters, no Devanagari/CJK/Thai/…).

    Under dual-query this is the whole trigger: transliteration is attempted only for
    Latin-script queries. A native-Devanagari query already embeds well and skips the path; a
    CJK/Thai query is not romanized Indic and skips it too. Deciding by *script* (not by
    guessing the language) is why there are no false positives — an English query is still
    dual-queried, but its raw form out-scores its transliteration, so nothing is harmed.
    """
    return bool(_LATIN_LETTER.search(text)) and not _NON_LATIN.search(text)
