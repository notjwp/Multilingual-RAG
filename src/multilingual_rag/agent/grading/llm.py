"""Opt-in relevance grading: ask the model whether the passages answer the question.

**Measured warning before you enable this.** With ``meta/llama-3.1-8b-instruct`` (the default
generation model) it is not usable as a judge: on XQuAD-hi it graded **81% of *correct* retrievals
as weak** (13 of 16), fired on 19/20 queries, and cost a provider call every time to be wrong four
times in five. In isolation it does better — asked about a single gold passage it answered
correctly 7/8 — but the real task is picking the relevant passage out of five mixed ones, and it
fails that. A larger judge model may well work; this one does not, which is why the default is
``score-threshold``.

Costs one extra provider call per retrieval attempt — on a credit-based free tier a
grade-and-retry turn can reach five calls. Enable with ``RELEVANCE_GRADER=llm``.

**Fails open.** An unparseable verdict, or any ``OpenAIError``, grades the retrieval as relevant.
A flaky judge must never block an answer the user could have had — the worst case of failing open
is one ordinary un-retried answer, while the worst case of failing closed is a pointless retry
loop ending in "I couldn't find it" over context that was fine.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAIError

from multilingual_rag.agent.grading.base import Grade
from multilingual_rag.core.models import VectorSearchResult
from multilingual_rag.generation.base import StreamClient

GRADER_SYSTEM = (
    "You judge whether retrieved passages contain the information needed to answer a question. "
    "Answer with exactly one word: YES if they do, NO if they do not. "
    "Judge only relevance to the question — not writing quality, completeness, or language."
)

# Long enough that the answer is actually *in* what the judge sees. This started at 400 and that
# was a bug: 88% of XQuAD-hi passages are longer than that (median 658, p75 841), so the evidence
# was being truncated away and the model then — correctly — reported that the passages did not
# contain the answer. It graded gold documents "NO" about half the time and the repair loop fired
# on 12/12 queries. 1500 covers essentially the whole corpus; at top_k=8 that is ~3k tokens, which
# is nothing against a 128k window.
_SNIPPET_CHARS = 1500


def build_grader_prompt(query: str, results: Sequence[VectorSearchResult]) -> str:
    """Format the question and truncated passages into the grading prompt."""
    passages = "\n\n".join(
        f"[{index}] {result.text[:_SNIPPET_CHARS]}"
        for index, result in enumerate(results, start=1)
    )
    return f"Question:\n{query}\n\nRetrieved passages:\n{passages}\n\nRelevant (YES/NO):"


class LlmRelevanceGrader:
    """Grade with a one-word LLM call, reusing the graph's existing stream client."""

    def __init__(self, *, client: StreamClient, model: str) -> None:
        self._client = client
        self._model = model

    async def grade(self, *, query: str, results: Sequence[VectorSearchResult]) -> Grade:
        """Ask the model whether ``results`` answer ``query``; fail open on any trouble."""
        top_score = max((result.score for result in results), default=None)
        if not results:
            # No call needed, and no judgement to make — this one is unambiguous.
            return Grade(relevant=False, reason="nothing was retrieved", top_score=None)

        try:
            raw = await self._client.acomplete(
                model=self._model,
                system=GRADER_SYSTEM,
                prompt=build_grader_prompt(query, results),
            )
        except OpenAIError:
            return Grade(relevant=True, reason="grader unavailable", top_score=top_score)

        verdict = raw.strip().strip(".").upper()
        if verdict.startswith("NO"):
            return Grade(relevant=False, reason="model judged the passages off-topic",
                         top_score=top_score)
        if verdict.startswith("YES"):
            return Grade(relevant=True, reason="model judged the passages relevant",
                         top_score=top_score)
        return Grade(relevant=True, reason="grader returned no clear verdict", top_score=top_score)
