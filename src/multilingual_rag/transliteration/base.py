"""Transliteration provider contract (a port, like ``EmbeddingProvider``)."""

from __future__ import annotations

from typing import Protocol


class Transliterator(Protocol):
    """Convert romanized (Latin-script) text into a target native script."""

    def transliterate(self, text: str, *, target_language: str) -> str:
        """Return ``text`` transliterated into ``target_language``'s native script.

        Implementations return the input unchanged when the language is unsupported, so a
        caller can transliterate unconditionally without a capability check.
        """
        ...
