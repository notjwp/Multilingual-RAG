"""Parse the citation markers a grounded answer actually used.

Context chunks are numbered ``[1] [2] …`` by ``retrieval.context.format_context`` and the
system prompt asks the model to cite supporting chunks by their bracket number. This maps those
markers back to the retrieved results so an answer cites only what it drew on — never the whole
retrieval set.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from multilingual_rag.core.models import AnswerCitation, VectorSearchResult

_MARKER = re.compile(r"\[(\d+)\]")


def parse_cited_results(
    answer: str,
    results: Sequence[VectorSearchResult],
) -> tuple[VectorSearchResult, ...]:
    """Return the retrieved results the answer cites, in first-seen order.

    Markers are 1-based (``[1]`` is ``results[0]``). Out-of-range markers are ignored, repeats
    are de-duplicated, and an answer with no valid markers cites nothing (never everything).
    """
    seen: set[int] = set()
    cited: list[VectorSearchResult] = []
    for raw in _MARKER.findall(answer):
        index = int(raw) - 1
        if 0 <= index < len(results) and index not in seen:
            seen.add(index)
            cited.append(results[index])
    return tuple(cited)


def answer_citations(
    answer: str,
    results: Sequence[VectorSearchResult],
) -> tuple[AnswerCitation, ...]:
    """Map the results an answer cites to ``AnswerCitation``s (the answer's citation set).

    Shared by the blocking and streaming generators so both produce citations the same way.
    """
    return tuple(
        AnswerCitation(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source=result.source,
            page=result.page,
            text=result.text,
        )
        for result in parse_cited_results(answer, results)
    )
