"""LLM roman->native transliteration (opt-in; costs generation credits).

Reuses the same OpenAI-compatible ``ChatClient`` as answer generation. The original spike measured
this as the strongest path (0.747 retention there), but that figure predates the romanization fix
in ``evaluation/romanization.py`` and has not been re-derived. It also adds a model call + credits
+ latency to *every* romanized query, so it is opt-in (``TRANSLITERATION_PROVIDER=llm``) and never
the default under the "free, ever" constraint.
"""

from __future__ import annotations

from multilingual_rag.generation.openai_compatible_generator import ChatClient

_LANGUAGE_NAME: dict[str, str] = {
    "hi": "Hindi (Devanagari script)",
    "kn": "Kannada script",
    "te": "Telugu script",
}

_SYSTEM = (
    "You transliterate romanized text into a native Indic script. "
    "Output ONLY the transliterated text — no quotes, no explanation, no romanization."
)


class LlmTransliterator:
    """Transliterate via an OpenAI-compatible chat model. Satisfies ``Transliterator``."""

    def __init__(self, *, client: ChatClient, model: str) -> None:
        self._client = client
        self._model = model

    def transliterate(self, text: str, *, target_language: str) -> str:
        language_name = _LANGUAGE_NAME.get(target_language)
        if language_name is None or not text.strip():
            return text
        prompt = f"Transliterate this romanized text into {language_name}:\n{text}"
        result = self._client.create_completion(
            model=self._model, system=_SYSTEM, prompt=prompt
        ).strip()
        return result or text
