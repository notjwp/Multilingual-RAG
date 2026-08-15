"""The agentic RAG graph: topology, plus the facade the API layer holds.

```
START ─┬─(has history)─► condense ─┐
       └──(no history)─────────────┴─► route_language ─► retrieve ─► grade
                                                            ▲           │
                                                            │  relevant │─► generate ─► END
                                                         repair ◄───────┤ weak + retries
                                                                        └─► generate_no_context
```

Nothing outside this module imports LangGraph — routes and the chat service depend on
:class:`RagGraph`, whose two methods are the whole surface.

**No tools and no checkpointer, both deliberate.** There is no ``ToolNode`` and no ``bind_tools``:
retrieval is a node, so the model never authors arguments and cannot reach another user's
documents by writing a different ``user_id``. And conversation history already lives in the
``messages`` table (loaded by ``ChatService._history``), so a LangGraph checkpointer would be a
second, competing source of truth for the same data.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import cast

from fastapi import status
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from multilingual_rag.agent.events import AgentEvent
from multilingual_rag.agent.nodes import RagNodes
from multilingual_rag.agent.state import AgentResult, AgentState, AgentUpdate
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import ConversationTurn, GeneratedAnswer, RetrievalContext
from multilingual_rag.vectorstores.base import VectorFilter

CompiledRagGraph = CompiledStateGraph[AgentState, Runtime[None], AgentState, AgentUpdate]

# A happy path is six super-steps (entry branch, condense, route, retrieve, grade, generate);
# each repair adds three (repair, retrieve, grade). The slack absorbs the entry branch and END.
_RECURSION_SLACK = 8
_STEPS_PER_REPAIR = 3


def build_graph(nodes: RagNodes) -> CompiledRagGraph:
    """Wire the nodes into the compiled graph. Pure topology — no Settings, no adapters."""
    builder: StateGraph[AgentState, Runtime[None], AgentState, AgentUpdate] = StateGraph(
        AgentState
    )
    builder.add_node("condense", nodes.condense)
    builder.add_node("route_language", nodes.route_language)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("repair", nodes.repair)
    builder.add_node("generate", nodes.generate)
    builder.add_node("generate_no_context", nodes.generate_no_context)

    builder.add_conditional_edges(
        START,
        nodes.needs_condense,
        {"condense": "condense", "route_language": "route_language"},
    )
    builder.add_edge("condense", "route_language")
    builder.add_edge("route_language", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        nodes.after_grade,
        {
            "generate": "generate",
            "repair": "repair",
            "generate_no_context": "generate_no_context",
        },
    )
    builder.add_edge("repair", "retrieve")  # the cycle
    builder.add_edge("generate", END)
    builder.add_edge("generate_no_context", END)
    return builder.compile()


class RagGraph:
    """The agentic RAG orchestration, as one object the API layer can hold.

    Replaces ``RagQueryService.answer_query``, ``RagQueryService.answer``, and
    ``StreamingAnswerGenerator.stream`` — three orchestrations, two implementations of the same
    seven steps, now one graph.
    """

    def __init__(self, graph: CompiledRagGraph, *, max_repairs: int) -> None:
        self._graph = graph
        self._recursion_limit = _STEPS_PER_REPAIR * max_repairs + _RECURSION_SLACK

    async def answer(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        top_k: int | None = None,
        filters: VectorFilter | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AgentResult:
        """Run the graph to completion and return the answer with its retrieval context."""
        payload = self._initial_state(
            query,
            user_id=user_id,
            session_id=session_id,
            preferred_language=preferred_language,
            top_k=top_k,
            filters=filters,
            history=history,
        )
        try:
            final = await self._graph.ainvoke(payload, config=self._config())
        except GraphRecursionError as exc:
            raise self._recursion_error() from exc

        answer, context = final["answer"], final["context"]
        # Narrows Any for mypy (warn_return_any) and asserts a real invariant in one move.
        if not isinstance(answer, GeneratedAnswer) or not isinstance(context, RetrievalContext):
            raise AppError(
                "The agent graph finished without an answer.",
                code="agent_incomplete",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return AgentResult(answer=answer, context=context)

    async def stream(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[AgentEvent]:
        """Yield agent steps and answer tokens as the graph runs, then the assembled answer."""
        payload = self._initial_state(
            query,
            user_id=user_id,
            session_id=session_id,
            preferred_language=preferred_language,
            top_k=None,
            filters=None,
            history=history,
        )
        try:
            async for chunk in self._graph.astream(
                payload, config=self._config(), stream_mode="custom"
            ):
                # The one deserialization boundary: the custom channel is untyped, and the only
                # thing any node writes to it is an AgentEvent (see agent/events.py::emit).
                yield cast(AgentEvent, chunk)
        except GraphRecursionError as exc:
            raise self._recursion_error() from exc

    def _config(self) -> RunnableConfig:
        return {"recursion_limit": self._recursion_limit}

    @staticmethod
    def _recursion_error() -> AppError:
        """An AppError, not a bare raise: ``chat_stream.py`` only renders ``event: error`` for
        AppError, so an unmapped GraphRecursionError would truncate the SSE stream silently."""
        return AppError(
            "The agent could not settle on an answer.",
            code="agent_recursion_limit",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @staticmethod
    def _initial_state(
        query: str,
        *,
        user_id: str,
        session_id: str | None,
        preferred_language: str | None,
        top_k: int | None,
        filters: VectorFilter | None,
        history: Sequence[ConversationTurn],
    ) -> AgentState:
        normalized = query.strip()
        return AgentState(
            question=normalized,
            user_id=user_id,
            session_id=session_id,
            history=tuple(history),
            preferred_language=preferred_language,
            top_k=top_k,
            filters=filters,
            search_query=normalized,
            route=None,
            context=None,
            grade=None,
            attempts=0,
            tried_strategies=(),
            answer=None,
        )
