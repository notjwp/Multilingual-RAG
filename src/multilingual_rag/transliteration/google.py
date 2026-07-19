"""Google Translate transliteration via googletrans (best quality; network + unofficial).

Uses the ``src="en"`` trick: telling Google the romanized Hindi is *English* makes it render the
same words in Devanagari (transliteration) instead of no-op'ing a hi->hi "translation". It was
the highest-quality option measured (near-exact on the sample) and is free — no API key.

Tradeoffs it accepts: it calls ``translate.googleapis.com`` per query, so it is **not offline**,
and googletrans is an unofficial scraper that breaks when Google shifts its endpoint. It is
therefore paired with a local rule-based fallback: any network/endpoint failure (or an empty
result) degrades to rule-based rather than hard-failing, and under dual-query a wrong
transliteration only ever *adds* a losing search, never replaces the raw one.
"""

from __future__ import annotations

import asyncio
import logging

from multilingual_rag.transliteration.base import Transliterator
from multilingual_rag.transliteration.rule_based import SanscriptTransliterator

logger = logging.getLogger(__name__)

# Google language codes for the transliteration target. Hindi ships; kn/te wired.
_DEST_CODE: dict[str, str] = {"hi": "hi", "kn": "kn", "te": "te"}


class GoogleTransliterator:
    """Transliterate via googletrans, falling back to rule-based. Satisfies ``Transliterator``."""

    def __init__(self, *, timeout: float = 5.0, fallback: Transliterator | None = None) -> None:
        self._timeout = timeout
        self._fallback = fallback or SanscriptTransliterator()

    def transliterate(self, text: str, *, target_language: str) -> str:
        dest = _DEST_CODE.get(target_language)
        if dest is None or not text.strip():
            return text
        try:
            result = asyncio.run(self._translate(text, dest))
        except Exception:
            logger.warning(
                "googletrans transliteration failed; using rule-based fallback", exc_info=True
            )
            result = ""
        return result or self._fallback.transliterate(text, target_language=target_language)

    async def _translate(self, text: str, dest: str) -> str:
        import httpx
        from googletrans import Translator  # type: ignore[import-untyped]

        async with Translator(
            raise_exception=True, timeout=httpx.Timeout(self._timeout)
        ) as translator:
            # src="en" forces transliteration of romanized Indic rather than a hi->hi no-op.
            translated = await translator.translate(text, src="en", dest=dest)
        return str(translated.text or "").strip()
