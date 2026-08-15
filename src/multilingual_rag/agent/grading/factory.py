"""Select the relevance grader from settings (the ``build_*`` factory convention)."""

from __future__ import annotations

from multilingual_rag.agent.grading.base import RelevanceGrader
from multilingual_rag.agent.grading.llm import LlmRelevanceGrader
from multilingual_rag.agent.grading.score_threshold import ScoreThresholdGrader
from multilingual_rag.core.config import Settings
from multilingual_rag.generation.base import StreamClient


def build_relevance_grader(settings: Settings, *, client: StreamClient) -> RelevanceGrader:
    """Return the grader named by ``RELEVANCE_GRADER``.

    ``client`` is the graph's existing stream client — only the llm grader uses it, but taking it
    unconditionally keeps the call site free of a branch.
    """
    if settings.relevance_grader == "llm":
        return LlmRelevanceGrader(client=client, model=settings.generation_model)
    return ScoreThresholdGrader(threshold=settings.relevance_score_threshold)
