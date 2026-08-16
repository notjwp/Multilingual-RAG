"""Grounding contract (a port, like ``RelevanceGrader`` and ``VectorStore``)."""

from __future__ import annotations

from typing import Protocol


class GroundingJudge(Protocol):
    """Decide whether every claim in a drafted answer is supported by the context it was given.

    Distinct from :class:`~multilingual_rag.agent.grading.base.RelevanceGrader`, which runs
    *before* generation and judges the retrieval. This runs *after*, and judges the answer — the
    two catch different failures. The 61% hallucination rate on unanswerable questions is a case
    the relevance grader passes (the passages look plausible) and only a grounding check can see.

    Sync, unlike ``RelevanceGrader``: the implementation is ``LlmFaithfulnessJudge``, which
    predates the graph and speaks the blocking ``ChatClient``. The node bridges with
    ``asyncio.to_thread`` rather than forking it into a second async copy — one judge and one
    prompt, shared by the offline eval and the live graph.
    """

    def is_supported(self, *, answer: str, context: str) -> bool:
        """Return True when every claim in ``answer`` is grounded in ``context``."""
        ...
