"""Select the transliteration adapter from settings.

Default is ``google`` (highest measured quality, free, but network-dependent) with a local
rule-based fallback baked into the adapter, so a network/endpoint failure degrades gracefully.
``indicxlit`` is the local/offline neural option; ``rule-based`` the zero-dependency floor;
``llm`` reuses the generation endpoint (costs credits). Returns ``None`` when disabled.
"""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.transliteration.base import Transliterator


def build_transliterator(settings: Settings) -> Transliterator | None:
    """Return the configured transliterator, or ``None`` when transliteration is disabled."""
    if not settings.transliteration_enabled:
        return None

    provider = settings.transliteration_provider
    if provider == "off":
        return None

    if provider == "rule-based":
        from multilingual_rag.transliteration.rule_based import SanscriptTransliterator

        return SanscriptTransliterator()

    if provider == "indicxlit":
        from multilingual_rag.transliteration.indicxlit import IndicXlitTransliterator

        return IndicXlitTransliterator(device=settings.embedding_device)

    if provider == "llm":
        from multilingual_rag.generation.openai_compatible_generator import (
            OpenAICompatibleChatClient,
        )
        from multilingual_rag.transliteration.llm import LlmTransliterator

        api_key = settings.generation_api_key
        if api_key is None:
            # No key to reach the LLM — degrade to the guaranteed-local transliterator.
            from multilingual_rag.transliteration.rule_based import SanscriptTransliterator

            return SanscriptTransliterator()
        client = OpenAICompatibleChatClient(
            api_key.get_secret_value(),
            settings.generation_base_url,
            settings.generation_timeout_seconds,
        )
        return LlmTransliterator(client=client, model=settings.generation_model)

    # Default: google, with its built-in rule-based fallback.
    from multilingual_rag.transliteration.google import GoogleTransliterator

    return GoogleTransliterator()
