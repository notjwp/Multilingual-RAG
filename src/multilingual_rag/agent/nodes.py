"""The agent graph's nodes: the actual work, one coroutine per step.

A class rather than free functions so the dependencies arrive by keyword-only constructor
injection — the same shape as ``RetrievalService`` and ``ChatService`` — and so every node can be
unit-tested directly (``await nodes.retrieve(state)``) with no graph involved. ``graph.py`` stays
pure topology.

**Blocking work is offloaded.** Retrieval is the sync RAG core (local bge-m3 embed + Chroma), and
so is routing: ``detect_target_language`` calls ``asyncio.run`` internally on the ``google``
detector path, which raises ``RuntimeError`` if invoked on a running event loop. Every call into
either goes through ``asyncio.to_thread``. The default ``word-list`` detector takes no such path,
so a regression here would pass the test suite and only fail in production — hence the explicit
test that routing runs off the main thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from fastapi import status
from openai import OpenAIError

from multilingual_rag.agent.events import Done, Step, Token, emit
from multilingual_rag.agent.grading.base import Grade, RelevanceGrader
from multilingual_rag.agent.repair import (
    REPAIR_REWRITE_SYSTEM,
    RepairStrategy,
    build_repair_rewrite_prompt,
    choose_repair,
    language_name,
    next_language,
)
from multilingual_rag.agent.state import AgentState, AgentUpdate
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import (
    GeneratedAnswer,
    RetrievalContext,
    VectorSearchResult,
)
from multilingual_rag.generation.base import StreamClient
from multilingual_rag.generation.citations import answer_citations, strip_unresolvable_markers
from multilingual_rag.generation.contextualize import (
    CONTEXTUALIZE_SYSTEM,
    build_contextualize_prompt,
    clean_standalone_query,
)
from multilingual_rag.generation.language import resolve_answer_language
from multilingual_rag.generation.openai_compatible_generator import generation_app_error
from multilingual_rag.generation.prompts import (
    NO_CONTEXT_SYSTEM,
    SYSTEM_INSTRUCTIONS,
    build_answer_prompt,
    build_no_context_prompt,
)
from multilingual_rag.retrieval.base import Retriever

GradeRoute = Literal["generate", "repair", "generate_no_context"]
EntryRoute = Literal["condense", "route_language"]


def _is_better(new: Grade, best: Grade | None) -> bool:
    """Should ``new`` replace the best attempt so far?

    Only when it clears the bar that the incumbent failed. Deliberately **not** "higher top score
    wins": comparing a transliterated search against a raw romanized one by cosine is the exact
    *relative* judgement ``transliteration/detect.py`` records as unreliable — "the raw romanized
    search finds enough high-cosine noise to look confident". Measured here: score-based selection
    scored 0.750 recall@5 against 0.800 for the plain pipeline, i.e. it lost, reproducing that
    finding. Using only the absolute relevant/weak verdict keeps the repair strictly opt-in — a
    retry has to actually solve the problem to be adopted, and a tie always keeps the original.
    """
    if best is None:
        return True
    return new.relevant and not best.relevant


class RagNodes:
    """The seven nodes and two edge routers of the agentic RAG graph."""

    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever,
        client: StreamClient,
        grader: RelevanceGrader,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.client = client
        self.grader = grader
        self.model = settings.generation_model

    # ── routers (pure; they may read state but never write it) ──────────────

    def needs_condense(self, state: AgentState) -> EntryRoute:
        """A first turn has nothing to condense — skip straight to routing.

        A conditional *entry* edge rather than an early return inside the node, so the two entry
        paths are visible in ``draw_mermaid()`` and "no LLM call on a first turn" is a structural
        property of the graph rather than a branch someone could delete.
        """
        return "condense" if state["history"] else "route_language"

    def after_grade(self, state: AgentState) -> GradeRoute:
        """Answer, repair-and-retry, or give up honestly."""
        grade = state["grade"]
        if grade is not None and grade.relevant:
            return "generate"
        if len(state["tried_strategies"]) >= self.settings.agent_max_repairs:
            return "generate_no_context"
        if self._next_strategy(state) is None:
            return "generate_no_context"
        return "repair"

    def _next_strategy(self, state: AgentState) -> RepairStrategy | None:
        """Recomputed in both the router and the repair node — it is pure and cheap, and routers
        cannot write to state."""
        return choose_repair(
            route=state["route"],
            question=state["question"],
            configured_languages=self.settings.transliteration_languages,
            tried=state["tried_strategies"],
        )

    # ── nodes ───────────────────────────────────────────────────────────────

    async def condense(self, state: AgentState) -> AgentUpdate:
        """Rewrite a follow-up into a standalone query so it embeds on its own."""
        emit(
            Step(
                id="condense",
                node="condense",
                status="running",
                label="Understanding your question",
            )
        )
        question = state["question"]
        try:
            raw = await self.client.acomplete(
                model=self.model,
                system=CONTEXTUALIZE_SYSTEM,
                prompt=build_contextualize_prompt(state["history"], question),
            )
        except OpenAIError as exc:
            raise generation_app_error(exc, self.model) from exc

        search_query = clean_standalone_query(raw, fallback=question)
        emit(
            Step(
                id="condense",
                node="condense",
                status="done",
                label="Understanding your question",
                detail=None if search_query == question else f"read as: {search_query}",
            )
        )
        return {"search_query": search_query}

    async def route_language(self, state: AgentState) -> AgentUpdate:
        """Decide which script to search in — the romanized-Indic decision, as its own step."""
        route = await asyncio.to_thread(self.retriever.route, state["search_query"])
        if route.transliteration_applied:
            emit(
                Step(
                    id="route_language",
                    node="route_language",
                    status="done",
                    label="Recognizing the language",
                    detail=f"{language_name(route.target_language)}, typed in English letters",
                )
            )
        return {"route": route}

    async def retrieve(self, state: AgentState) -> AgentUpdate:
        """Search this chat's documents. Scoped by ``user_id`` *and* ``session_id``, always."""
        attempt = state["attempts"] + 1
        step_id = f"retrieve:{attempt}"
        label = "Searching your documents" if attempt == 1 else "Searching again"
        emit(Step(id=step_id, node="retrieve", status="running", label=label))

        context = await asyncio.to_thread(
            self.retriever.retrieve,
            state["search_query"],
            user_id=state["user_id"],
            session_id=state["session_id"],
            top_k=state["top_k"],
            filters=state["filters"],
            route=state["route"],
        )
        found = len(context.results)
        emit(
            Step(
                id=step_id,
                node="retrieve",
                status="done",
                label=label,
                detail=f"{found} passage{'' if found == 1 else 's'} found",
            )
        )
        return {"context": context, "attempts": attempt}

    async def grade(self, state: AgentState) -> AgentUpdate:
        """Judge whether the retrieval is worth answering from, and keep the best attempt.

        Emits no step of its own: a passing grade is invisible to the user by design, and a
        failing one is announced by the repair step that follows.

        **Never regress.** Generation answers from ``best_context``, not the latest one. A repair
        is a bet, and it can lose: falling back to the raw romanized query is right only when the
        transliterated search genuinely failed, and no grader separates those perfectly — the
        score bands for correct and incorrect retrievals overlap (measured on XQuAD-hi: hits reach
        down to 0.42, misses up to 0.46). Without this, a low-scoring *correct* retrieval gets
        replaced by a worse one, and the agent scores below the plain pipeline. It did, before
        this existed: recall@5 0.733 agentic vs 0.800 shipped. Now a repair can only help or tie.
        """
        context = state["context"]
        grade = await self.grader.grade(
            query=state["search_query"],
            results=context.results if context is not None else (),
        )
        update: AgentUpdate = {"grade": grade}
        if context is not None and _is_better(grade, state["best_grade"]):
            update["best_grade"] = grade
            update["best_context"] = context
        return update

    async def repair(self, state: AgentState) -> AgentUpdate:
        """Change something about the search, then let the graph retry retrieval."""
        strategy = self._next_strategy(state)
        if strategy is None:  # pragma: no cover — after_grade routes elsewhere first
            return {}

        emit(
            Step(
                id=f"repair:{len(state['tried_strategies']) + 1}",
                node="repair",
                status="running",
                label="Didn't find much — trying again",
            )
        )
        update: AgentUpdate = {}
        detail = ""

        if strategy == "raw_fallback":
            route = await asyncio.to_thread(
                self.retriever.route, state["search_query"], skip_transliteration=True
            )
            update["route"] = route
            detail = "searching your original wording instead"
        elif strategy == "relanguage":
            current = state["route"].target_language if state["route"] else None
            target = next_language(self.settings.transliteration_languages, current)
            route = await asyncio.to_thread(
                self.retriever.route, state["search_query"], force_language=target
            )
            update["route"] = route
            detail = f"trying {language_name(target)} instead"
        else:  # rewrite
            try:
                raw = await self.client.acomplete(
                    model=self.model,
                    system=REPAIR_REWRITE_SYSTEM,
                    prompt=build_repair_rewrite_prompt(state["search_query"]),
                )
            except OpenAIError as exc:
                raise generation_app_error(exc, self.model) from exc
            rewritten = clean_standalone_query(raw, fallback=state["search_query"])
            update["search_query"] = rewritten
            update["route"] = await asyncio.to_thread(self.retriever.route, rewritten)
            detail = "rephrasing the search"

        emit(
            Step(
                id=f"repair:{len(state['tried_strategies']) + 1}",
                node="repair",
                status="done",
                label="Didn't find much — trying again",
                detail=detail,
            )
        )
        update["tried_strategies"] = state["tried_strategies"] + (strategy,)
        return update

    async def generate(self, state: AgentState) -> AgentUpdate:
        """Stream a grounded, cited answer from the retrieved context."""
        context = self._require_context(state)
        if context.query != state["question"]:
            # Retrieval used the condensed/repaired query; answer the user's actual wording.
            context = context.model_copy(update={"query": state["question"]})
        response_language = self._response_language(state, context, context.results)
        prompt = build_answer_prompt(context, response_language=response_language)
        answer_text = await self._stream_answer(
            system=SYSTEM_INSTRUCTIONS, prompt=prompt, state=state
        )
        # Drop markers that resolve to nothing before the text reaches the client, or the UI
        # renders a superscript citation with no matching source (see citations.py).
        answer = GeneratedAnswer(
            answer=strip_unresolvable_markers(answer_text, context.results),
            language=response_language,
            citations=answer_citations(answer_text, context.results),
        )
        emit(Done(answer))
        return {"answer": answer, "context": context}

    async def generate_no_context(self, state: AgentState) -> AgentUpdate:
        """Say plainly that the documents don't cover it, with guaranteed-empty citations."""
        context = self._require_context(state)
        response_language = self._response_language(state, context, ())
        prompt = build_no_context_prompt(state["question"], response_language=response_language)
        answer_text = await self._stream_answer(
            system=NO_CONTEXT_SYSTEM, prompt=prompt, state=state
        )
        # No context, so every marker is unresolvable — strip them all rather than show a
        # refusal that appears to cite something.
        answer = GeneratedAnswer(
            answer=strip_unresolvable_markers(answer_text, ()),
            language=response_language,
            citations=(),
        )
        emit(Done(answer))
        return {"answer": answer, "context": context}

    # ── shared generation plumbing ──────────────────────────────────────────

    @staticmethod
    def _response_language(
        state: AgentState, context: RetrievalContext, results: Sequence[VectorSearchResult]
    ) -> str:
        """Pick the answer language, trusting the router over langdetect.

        ``langdetect`` guesses from Latin script and gets romanized Indic badly wrong — it labels
        ``bharat ki rajdhani kya hai`` as Swahili. ``route_language`` has already identified the
        real language with the purpose-built detector (~98% accurate), so its verdict wins.

        This surfaced live: a no-context refusal for a romanized Hindi query came back in
        *Albanian*. The normal path masks the same bug — with Devanagari passages in the prompt the
        model mirrors them and ignores the bad hint — so it only became visible once retrieval
        returned nothing. Fixed for both paths, since the hint was wrong in both.
        """
        route = state["route"]
        detected = route.target_language if route is not None else None
        return resolve_answer_language(
            state["preferred_language"], detected or context.query_language, results
        )

    async def _stream_answer(self, *, system: str, prompt: str, state: AgentState) -> str:
        """Stream the model's reply, emitting tokens as they arrive, and return the whole text.

        Always streams, even for the blocking routes: under ``ainvoke`` ``emit`` is a no-op and
        this simply accumulates. That is why generation no longer needs a blocking twin.
        """
        emit(Step(id="generate", node="generate", status="running", label="Writing the answer"))
        parts: list[str] = []
        try:
            async for delta in self.client.astream_completion(
                model=self.model, system=system, prompt=prompt, history=state["history"]
            ):
                parts.append(delta)
                emit(Token(delta))
        except OpenAIError as exc:
            raise generation_app_error(exc, self.model) from exc

        answer_text = "".join(parts).strip()
        if not answer_text:
            raise AppError(
                "The generation endpoint returned an empty answer.",
                code="empty_generation_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
        emit(Step(id="generate", node="generate", status="done", label="Writing the answer"))
        return answer_text

    @staticmethod
    def _require_context(state: AgentState) -> RetrievalContext:
        """The attempt to answer from: the best one seen, not necessarily the last.

        Also narrows for mypy and asserts the invariant that retrieve ran first.
        """
        context = state["best_context"] or state["context"]
        if context is None:  # pragma: no cover — retrieve always precedes generation
            raise AppError(
                "The agent reached generation without retrieving.",
                code="agent_incomplete",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return context
