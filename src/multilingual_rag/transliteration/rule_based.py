"""Rule-based roman->native transliteration via indic-transliteration (sanscript).

The guaranteed-local fallback: no model download, instant, free. Quality is limited on
free-form input (ITRANS-family schemes expect explicit long-vowel markers like ``aa`` that
people omit when typing), but under dual-query it can only *add* hits — a mangled-but-real
Devanagari form still shares the consonant skeleton with the native text, which bge-m3 matches
far better than Latin script does.
"""

from __future__ import annotations

from indic_transliteration import sanscript  # type: ignore[import-untyped]
from indic_transliteration.sanscript import transliterate  # type: ignore[import-untyped]

# Hindi ships now; kn/te are wired for when their eval sets exist.
_TARGET_SCRIPT: dict[str, str] = {
    "hi": sanscript.DEVANAGARI,
    "kn": sanscript.KANNADA,
    "te": sanscript.TELUGU,
}


class SanscriptTransliterator:
    """Transliterate with a fixed sanscript scheme. Satisfies ``Transliterator``."""

    def __init__(self, source_scheme: str = sanscript.ITRANS) -> None:
        self._source_scheme = source_scheme

    def transliterate(self, text: str, *, target_language: str) -> str:
        target = _TARGET_SCRIPT.get(target_language)
        if target is None or not text.strip():
            return text
        return str(transliterate(text, self._source_scheme, target))
