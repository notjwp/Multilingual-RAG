"""The agent graph's working memory for one turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from multilingual_rag.agent.grading.base import Grade
from multilingual_rag.agent.repair import RepairStrategy
from multilingual_rag.core.models import ConversationTurn, GeneratedAnswer, RetrievalContext
from multilingual_rag.retrieval.routing import LanguageRoute
from multilingual_rag.vectorstores.base import VectorFilter


class AgentState(TypedDict):
    """One turn's state. Inputs are written once by ``RagGraph``; the rest accumulate.

    ``user_id`` and ``session_id`` live here, injected from the authenticated request, and no
    node ever recomputes them. Because retrieval is a *node* rather than an LLM-callable tool,
    the model has no way to author them — the tenancy boundary is structural, not prompt-enforced.
    """

    # ── inputs: write-once, from the authenticated request ──────────────────
    question: str  # the user's literal wording, preserved for display and generation
    user_id: str
    session_id: str | None
    history: tuple[ConversationTurn, ...]
    preferred_language: str | None
    top_k: int | None
    filters: VectorFilter | None

    # ── accumulated ─────────────────────────────────────────────────────────
    search_query: str  # condense writes it; repair may rewrite it
    route: LanguageRoute | None
    context: RetrievalContext | None  # the most recent attempt
    grade: Grade | None  # the most recent verdict; the repair router reads this one
    attempts: int
    tried_strategies: tuple[RepairStrategy, ...]

    # The best attempt seen so far, which is what generation actually answers from. A repair can
    # make retrieval *worse* — falling back to the raw romanized query is a good bet only when the
    # transliterated search really did fail, and the grader cannot tell those apart perfectly. See
    # the "never regress" note in nodes.py::grade.
    best_context: RetrievalContext | None
    best_grade: Grade | None

    # ── terminal ────────────────────────────────────────────────────────────
    answer: GeneratedAnswer | None


class AgentUpdate(TypedDict, total=False):
    """The subset of :class:`AgentState` a node may write.

    Not decoration: LangGraph **silently ignores unknown keys** in a node's return dict, so a
    typo like ``{"contxt": ...}`` would never raise and the state would just stay stale. Typing
    node returns as this instead of ``dict[str, Any]`` makes mypy catch it.

    No reducers and no ``Annotated`` accumulators — the graph is strictly sequential, so
    last-write-wins is correct for every field. ``tried_strategies`` grows by the repair node
    returning ``state["tried_strategies"] + (strategy,)``.
    """

    search_query: str
    route: LanguageRoute
    context: RetrievalContext
    grade: Grade
    best_context: RetrievalContext
    best_grade: Grade
    attempts: int
    tried_strategies: tuple[RepairStrategy, ...]
    answer: GeneratedAnswer


@dataclass(frozen=True)
class AgentResult:
    """One completed turn. Carries the context too, because ``/v1/query`` reports the retrieved
    chunks, the detected query language, and the transliteration transparency fields."""

    answer: GeneratedAnswer
    context: RetrievalContext
