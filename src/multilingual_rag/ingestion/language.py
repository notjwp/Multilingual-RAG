"""Language detection utilities."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore[import-untyped]

DetectorFactory.seed = 0


class LanguageDetector:
    """Detect ISO 639-1 language codes using langdetect."""

    def __init__(self, min_text_length: int = 20) -> None:
        if min_text_length < 0:
            raise ValueError("min_text_length must be greater than or equal to zero")
        self.min_text_length = min_text_length

    def detect(self, text: str, *, default: str = "unknown") -> str:
        """Return a language code, falling back for empty or ambiguous text."""
        normalized_text = " ".join(text.split())
        if len(normalized_text) < self.min_text_length:
            return default

        try:
            return str(detect(normalized_text))
        except LangDetectException:
            return default
