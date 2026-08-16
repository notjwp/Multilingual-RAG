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
# Inline code / fenced blocks, so a literal "[1]" in a snippet is never treated as a citation.
_CODE_SPAN = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
# Same marker, but capturing any space in front of it so a deletion leaves no double space.
_MARKER_WITH_SPACE = re.compile(r"([ \t]*)\[(\d+)\]")


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


def strip_unresolvable_markers(answer: str, results: Sequence[VectorSearchResult]) -> str:
    """Remove citation markers that point at nothing, leaving the resolvable ones intact.

    ``parse_cited_results`` already ignores an out-of-range marker, but the *answer text* still
    contains it, so the UI renders a superscript citation with no matching source. Seen live: an
    answer written against 1 resolvable passage said "… swasthya hi dhan hai. [2]" and the reader
    got a dangling [2].

    Deliberately **does not renumber**. Rewriting ``[5]`` to ``[2]`` would invent an attribution
    the model never made; deleting an unresolvable marker is honest, remapping it is not.

    Markers inside inline code are left alone, matching the frontend's ``rehypeCitations``, which
    skips ``code``/``pre`` — the two must agree or the rendered output disagrees with the parse.
    """
    spans = [match.span() for match in _CODE_SPAN.finditer(answer)]

    def in_code(position: int) -> bool:
        return any(start <= position < end for start, end in spans)

    def replace(match: re.Match[str]) -> str:
        # The pattern swallows any leading space, so removing a marker cannot leave a double
        # space behind ("aur [5] mein" -> "aur mein"). Collapsing whitespace globally instead
        # would reformat code spans, which must survive byte-for-byte.
        if in_code(match.start()):
            return match.group(0)
        index = int(match.group(2)) - 1
        return match.group(0) if 0 <= index < len(results) else ""

    return _MARKER_WITH_SPACE.sub(replace, answer).strip()
