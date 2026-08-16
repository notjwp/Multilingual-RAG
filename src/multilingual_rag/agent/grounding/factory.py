"""Select the grounding judge from settings (the ``build_*`` factory convention)."""

from __future__ import annotations

from multilingual_rag.agent.grounding.base import GroundingJudge
from multilingual_rag.core.config import Settings
from multilingual_rag.evaluation.llm_judge import LlmFaithfulnessJudge


def build_grounding_judge(settings: Settings) -> GroundingJudge | None:
    """Return the post-generation judge, or ``None`` when the gate is off (the default).

    ``None`` rather than a no-op judge, so ``RagNodes`` can skip the whole node — a disabled gate
    must cost nothing, not a provider call that always says yes.

    The adapter is imported from ``evaluation/`` (the one direction production code depends on
    that package) deliberately: ``LlmFaithfulnessJudge`` and its ``JUDGE_SYSTEM`` prompt already
    exist and are already exercised by ``evaluation/run.py``. A second copy under ``agent/`` would
    be the same prompt scored two ways, and the offline number would stop predicting the live one.
    """
    if not settings.grounding_gate:
        return None
    return LlmFaithfulnessJudge(settings)
