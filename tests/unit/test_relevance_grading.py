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


def test_factory_defaults_to_the_free_score_threshold_grader_at_a_zero_floor() -> None:
    """Both halves of this default are measured, not cautious. The llm grader false-alarms on 81%
    of correct retrievals with llama-3.1-8b, and every positive cosine floor tried scored below
    having no agent at all (best 0.767 vs 0.800) because the score bands overlap. A 0.0 floor
    means only a *literally empty* retrieval is graded weak — the one signal cosine can be
    trusted on — so the repair loop cannot fire on a hunch."""
    grader = build_relevance_grader(Settings(), client=FakeStreamClient())

    assert isinstance(grader, ScoreThresholdGrader)
    assert grader.threshold == 0.0
    assert asyncio.run(grader.grade(query="q", results=(_result("c1", 0.01),))).relevant is True
    assert asyncio.run(grader.grade(query="q", results=())).relevant is False


def test_factory_selects_the_llm_grader_when_configured() -> None:
    grader = build_relevance_grader(
        Settings(relevance_grader="llm"), client=FakeStreamClient()
    )

    assert isinstance(grader, LlmRelevanceGrader)


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
