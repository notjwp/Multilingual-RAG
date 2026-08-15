"""Free relevance grading: did anything clear the cosine floor at all.

**On the apparent contradiction with** ``transliteration/detect.py``, which records that
"score-based routing proved unreliable at scale (the raw romanized search finds enough high-cosine
noise to look confident)". That finding stands, and it is not this.

- detect.py was **comparing two competing query forms** — a *relative* judgement across different
  embeddings of different scripts, asking which of two searches to trust. That is hard, and it lost.
- This grader makes an *absolute* abstain check: did anything at all clear the floor. Different
  question, different failure mode.

So the check is deliberately conservative and fails open — it fires on unambiguous misses only:

- ``not results`` → weak. No threshold involved; this case is 100% reliable.
- ``max(score) < threshold`` → weak.

**Known limitation, stated rather than hidden:** bge-m3 cosine scores are *not calibrated across
languages*. A Devanagari query and an English query have different score distributions against the
same corpus, so a single global floor is inherently more aggressive for one than the other. The
default is set low for that reason; ``RELEVANCE_SCORE_THRESHOLD`` is the knob.
"""

from __future__ import annotations

from collections.abc import Sequence

from multilingual_rag.agent.grading.base import Grade
from multilingual_rag.core.models import VectorSearchResult


class ScoreThresholdGrader:
    """Grade on the top cosine score. Zero LLM calls — the default, so demos stay free."""

    def __init__(self, *, threshold: float) -> None:
        self.threshold = threshold

    async def grade(self, *, query: str, results: Sequence[VectorSearchResult]) -> Grade:
        """Return weak when nothing was retrieved, or when the best match is below the floor."""
        if not results:
            return Grade(relevant=False, reason="nothing was retrieved", top_score=None)

        top_score = max(result.score for result in results)
        if top_score < self.threshold:
            return Grade(
                relevant=False,
                reason=f"best match scored {top_score:.2f}, below {self.threshold:.2f}",
                top_score=top_score,
            )
        return Grade(
            relevant=True,
            reason=f"{len(results)} passages, best {top_score:.2f}",
            top_score=top_score,
        )
