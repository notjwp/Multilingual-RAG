"""The default relevance grader: ask the model whether the passages answer the question.

This is the shipped default (``RELEVANCE_GRADER=llm``), chosen on refusal quality — it is the only
configuration measured that never fabricates a cited answer to an out-of-corpus question. It costs
one provider call per retrieval attempt, so a grade-and-retry turn reaches ~3 calls.

**It is a poor judge, and that is the accepted cost.** With ``meta/llama-3.1-8b-instruct`` it
grades **81% of *correct* retrievals weak** (13 of 16) and says NO to 17 of 20 queries — which is
the 70% false-refusal rate the product ships with. The same model asked about a *single* passage
is right 7/8; it fails at picking the relevant passage out of five mixed ones.

**Do not re-propose a selection prompt.** The obvious inference from that 7/8 — ask "which
passages help, reply with numbers or NONE" and play to the strength — was built, measured, and
reverted. It works in the direction predicted and still loses:

===========================  ==============  ====================  ==============
prompt                       false alarms    fabricates            refuses answerable
===========================  ==============  ====================  ==============
set-level YES/NO (shipped)   81% (13/16)     0%                    70%
selection, numbers or NONE   56% (9/16)      **20%**               50%
===========================  ==============  ====================  ==============

The reframing did not make the judge *more accurate*, it made it **more permissive** — it dropped
wrong weak grades and right ones together (catches fell 4/4 -> 3/4, which reads as noise at n=4 and
as four fabricated answers at n=20). Trading the 0% fabrication rate for halved refusals reverses
the reason this grader is the default at all. Reports:
``data/eval/reports/grader-llama31-8b-selection.json`` and ``refusal-llm-selection.json``.

So this judge's errors cannot be separated into "wrong refusals" and "right refusals" by
prompting; a stronger judge model is the only route left. Re-measure with
``scripts/eval_grader.py`` after touching the prompt — the bar is two-sided (false alarms <20% AND
catches >=50%) precisely because this grader fails open, so a model that errors on every call would
otherwise post a flawless false-alarm rate.

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
