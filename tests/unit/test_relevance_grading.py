import asyncio
from collections.abc import AsyncIterator, Sequence

from openai import APIError

from multilingual_rag.agent.grading.factory import build_relevance_grader
from multilingual_rag.agent.grading.llm import LlmRelevanceGrader, build_grader_prompt
from multilingual_rag.agent.grading.score_threshold import ScoreThresholdGrader
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import ConversationTurn, VectorSearchResult


def _result(chunk_id: str, score: float, text: str = "some passage text") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        language="en",
        source="doc.txt",
        chunk_index=0,
        score=score,
        token_count=3,
    )


class FakeStreamClient:
    """Returns a canned verdict, or raises, so the grader's failure paths are testable."""

    def __init__(self, verdict: str = "YES", *, raises: Exception | None = None) -> None:
        self._verdict = verdict
        self._raises = raises
        self.prompts: list[str] = []

    async def astream_completion(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        yield self._verdict

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        if self._raises is not None:
            raise self._raises
        self.prompts.append(prompt)
        return self._verdict


# --- ScoreThresholdGrader (the free default) ---------------------------------------------------


def test_empty_results_are_weak_without_consulting_the_threshold() -> None:
    grader = ScoreThresholdGrader(threshold=0.35)

    grade = asyncio.run(grader.grade(query="anything", results=()))

    assert grade.relevant is False
    assert grade.top_score is None
    assert grade.reason == "nothing was retrieved"


def test_a_top_score_below_the_floor_is_weak() -> None:
    grader = ScoreThresholdGrader(threshold=0.35)

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.2), _result("c2", 0.1))))

    assert grade.relevant is False
    assert grade.top_score == 0.2
    # The reason is shown to a user as the agent step's detail, so it must read as prose.
    assert "below" in grade.reason


def test_a_top_score_at_or_above_the_floor_is_relevant() -> None:
    grader = ScoreThresholdGrader(threshold=0.35)

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.35), _result("c2", 0.9))))

    assert grade.relevant is True
    assert grade.top_score == 0.9  # the best match, not the first


# --- LlmRelevanceGrader (opt-in) ---------------------------------------------------------------


def test_llm_grader_accepts_a_yes_verdict() -> None:
    grader = LlmRelevanceGrader(client=FakeStreamClient("YES"), model="m")

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.8),)))

    assert grade.relevant is True


def test_llm_grader_accepts_a_no_verdict() -> None:
    grader = LlmRelevanceGrader(client=FakeStreamClient("NO."), model="m")

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.8),)))

    assert grade.relevant is False
    assert grade.top_score == 0.8


def test_llm_grader_fails_open_on_an_unparseable_verdict() -> None:
    # A judge that rambles must not cost the user an answer they could have had.
    grader = LlmRelevanceGrader(client=FakeStreamClient("Well, it depends..."), model="m")

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.8),)))

    assert grade.relevant is True
    assert grade.reason == "grader returned no clear verdict"


def test_llm_grader_fails_open_when_the_provider_errors() -> None:
    boom = APIError("upstream exploded", request=None, body=None)  # type: ignore[arg-type]
    grader = LlmRelevanceGrader(client=FakeStreamClient(raises=boom), model="m")

    grade = asyncio.run(grader.grade(query="q", results=(_result("c1", 0.8),)))

    assert grade.relevant is True
    assert grade.reason == "grader unavailable"


def test_llm_grader_skips_the_call_entirely_when_nothing_was_retrieved() -> None:
    client = FakeStreamClient("YES")
    grader = LlmRelevanceGrader(client=client, model="m")

    grade = asyncio.run(grader.grade(query="q", results=()))

    assert grade.relevant is False
    assert client.prompts == []  # no credits spent on an unambiguous case


# --- factory -----------------------------------------------------------------------------------


def test_factory_defaults_to_the_llm_grader() -> None:
    """The default is chosen on refusal quality, not on retrieval metrics.

    ``score-threshold`` scores better on XQuAD and costs a third as many provider calls, and it
    was the default until the refusal eval existed. But XQuAD has no unanswerable questions, so
    it could not see that the cheap grader lets the model fabricate an answer — with a citation
    attached to an unrelated passage — 61% of the time it is asked something out of corpus. The
    llm grader takes that to 0%. It pays for it by refusing 70% of *answerable* questions, which
    is a bad trade whenever the documents usually do hold the answer; the two cross at roughly a
    55% answerable share. See docs/architecture.md §1.9a and data/eval/reports/refusal-*.json.
    """
    grader = build_relevance_grader(Settings(), client=FakeStreamClient())

    assert isinstance(grader, LlmRelevanceGrader)


def test_factory_selects_the_score_threshold_grader_when_configured() -> None:
    """The free escape hatch, for a corpus that usually holds the answer or a key with no
    headroom for ~3 calls per turn. A 0.0 floor grades only a *literally empty* retrieval weak —
    the one signal cosine can be trusted on, since the score bands for correct and incorrect
    retrievals overlap (correct 0.424-0.696, incorrect 0.389-0.462)."""
    grader = build_relevance_grader(
        Settings(relevance_grader="score-threshold"), client=FakeStreamClient()
    )

    assert isinstance(grader, ScoreThresholdGrader)
    assert grader.threshold == 0.0
    assert asyncio.run(grader.grade(query="q", results=(_result("c1", 0.01),))).relevant is True
    assert asyncio.run(grader.grade(query="q", results=())).relevant is False


# --- the prompt the judge actually sees ----------------------------------------------------------


def test_the_grader_prompt_does_not_truncate_a_realistic_passage() -> None:
    """Regression: the snippet cap was 400 chars while 88% of XQuAD passages are longer (median
    658). The evidence was being cut out before the judge saw it, so it correctly answered that
    the passages did not contain the answer — grading *gold* documents "NO" about half the time
    and firing the repair loop on 12/12 queries."""
    answer = "THE-ANSWER-IS-HERE"
    passage = ("filler " * 120) + answer  # ~860 chars, the answer at the very end

    prompt = build_grader_prompt("what is the answer?", (_result("c1", 0.5, text=passage),))

    assert answer in prompt


def test_the_grader_prompt_carries_the_question_and_numbers_the_passages() -> None:
    prompt = build_grader_prompt(
        "who founded it?", (_result("c1", 0.9, text="alpha"), _result("c2", 0.8, text="beta"))
    )

    assert "who founded it?" in prompt
    assert "[1] alpha" in prompt
    assert "[2] beta" in prompt


# --- the selection prompt that was tried and reverted --------------------------------------------


def test_the_grader_asks_for_a_set_level_verdict_not_a_passage_selection() -> None:
    """Pins the reverted experiment so it is not silently reintroduced.

    Asking "which passages help, reply with numbers or NONE" is the obvious inference from this
    model scoring 7/8 on single passages while failing on mixed sets. It was built and measured:
    false alarms improved 81% -> 56% and false refusals 70% -> 50%, but fabrication went
    **0% -> 20%**, because the reframing made the judge more *permissive* rather than more
    *accurate* — it shed right weak grades along with wrong ones. That trades away the one property
    this grader is the default for. See the module docstring and
    data/eval/reports/*-selection.json.
    """
    prompt = build_grader_prompt("q", (_result("c1", 0.9, text="alpha"),))

    assert "YES/NO" in prompt
