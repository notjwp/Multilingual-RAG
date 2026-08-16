"""Free relevance grading: did anything clear the cosine floor at all.

**On the apparent contradiction with** ``transliteration/detect.py``, which records that
"score-based routing proved unreliable at scale (the raw romanized search finds enough high-cosine
noise to look confident)". That finding stands, and it is not this.

- detect.py was **comparing two competing query forms** — a *relative* judgement across different
  embeddings of different scripts, asking which of two searches to trust. That is hard, and it lost.
- This grader makes an *absolute* abstain check: did anything at all clear the floor. Different
  question, different failure mode.

So the check is deliberately conservative and fires on unambiguous misses only:

- ``not results`` → weak. No threshold involved; this case is 100% reliable.
- ``max(score) < threshold`` → weak. **The threshold defaults to 0.0, so this arm is off.**

**Why the threshold defaults to off — measured, not assumed.** On XQuAD-hi (3240 docs, 60 queries)
the top-1 cosine bands for correct and incorrect retrievals overlap badly:

    attempt        min    p25    median   max
    hit  (n=49)    0.424  0.475  0.524    0.696
    miss (n=11)    0.389  0.415  0.431    0.462

At the best separating floor (0.45) the check fires on 14/60 queries and only 8 of those are real
misses — so it condemns 6 correct retrievals, and the raw-query fallback it triggers replaces them
with something worse. End to end that *lost*: recall@5 0.767 against 0.800 for the plain pipeline,
across three different rules for choosing between attempts. This is the same effect
``transliteration/detect.py`` records — "the raw romanized search finds enough high-cosine noise
to look confident" — and it does not go away by moving the number.

An absolute abstain check is still a different question from that *relative* one, and the empty
case answers it perfectly. So the free grader keeps only the arm it can defend. Real judgement
needs a judge: ``RELEVANCE_GRADER=llm``.
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
