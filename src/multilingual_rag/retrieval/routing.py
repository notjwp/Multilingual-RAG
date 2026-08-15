"""Decide which text to embed for a query — the transliteration decision, as a value.

Split out of ``RetrievalService.retrieve`` so the agent graph can make this decision as its own
step and, when a retrieval comes back weak, *re*-make it differently without re-detecting: the
graph passes the resulting :class:`LanguageRoute` straight into ``retrieve``.

``force_language`` and ``skip_transliteration`` exist for that second attempt. They are the whole
reason the split is worth doing — see ``agent/repair.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from multilingual_rag.core.config import Settings
from multilingual_rag.transliteration.base import Transliterator
from multilingual_rag.transliteration.detect import detect_target_language


@dataclass(frozen=True)
class LanguageRoute:
    """Which text to embed for a query, and why.

    ``target_language`` and ``transliterated_query`` are both None when the query was left
    untouched — including when transliteration was attempted but turned out to be a no-op.
    """

    search_text: str
    target_language: str | None
    transliterated_query: str | None

    @property
    def transliteration_applied(self) -> bool:
        """True when the search text differs from the query because of transliteration."""
        return self.transliterated_query is not None


def route_query(
    query: str,
    *,
    settings: Settings,
    transliterator: Transliterator | None,
    force_language: str | None = None,
    skip_transliteration: bool = False,
) -> LanguageRoute:
    """Return the route for ``query``: transliterated to a native script, or left as-is.

    ``force_language`` skips detection and transliterates to that language directly.
    ``skip_transliteration`` returns the raw query untouched — the repair path's cheapest retry,
    for when the transliterated search is the thing that failed.
    """
    if skip_transliteration or transliterator is None:
        return LanguageRoute(search_text=query, target_language=None, transliterated_query=None)

    target = force_language or detect_target_language(
        query,
        settings.transliteration_languages,
        detector=settings.transliteration_detector,
    )
    if target is None:
        return LanguageRoute(search_text=query, target_language=None, transliterated_query=None)

    transliterated = transliterator.transliterate(query, target_language=target)
    if not transliterated.strip() or transliterated.strip() == query.strip():
        # A no-op transliteration is not a route — searching it would be searching the raw query
        # while claiming otherwise in the response's transparency fields.
        return LanguageRoute(search_text=query, target_language=None, transliterated_query=None)

    return LanguageRoute(
        search_text=transliterated,
        target_language=target,
        transliterated_query=transliterated,
    )
