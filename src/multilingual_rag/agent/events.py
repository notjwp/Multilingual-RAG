"""Ephemeral events the agent graph streams out while it works.

Everything here is *transport*, not state: the graph pushes these onto LangGraph's ``custom``
stream channel, the chat service re-wraps them, and the SSE route writes them to the wire. Steps
are never persisted — reloading a chat shows the answer and its citations, nothing else.

**Why one custom channel rather than ``stream_mode="messages"``.** That mode taps LangChain's
callback manager for ``BaseChatModel`` output, and generation here is the raw ``openai`` SDK
against an OpenAI-compatible endpoint (see ``generation/openai_compatible_generator.py``), which
produces no such events. Adopting ``langchain-openai`` to unlock it would mean a second HTTP stack
alongside the ``openai`` SDK that ``evaluation/llm_judge.py`` and ``transliteration/`` still need,
plus re-deriving the tested error contract in ``generation_app_error``. So tokens and steps share
one channel and are discriminated by type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langgraph.config import get_stream_writer

from multilingual_rag.core.models import GeneratedAnswer

NodeName = Literal[
    "condense",
    "route_language",
    "retrieve",
    "grade",
    "repair",
    "generate",
    "ground_check",
]


@dataclass(frozen=True)
class Token:
    """One streamed piece of the answer text."""

    text: str


@dataclass(frozen=True)
class Step:
    """One agent step, shown live in the chat UI and then collapsed.

    ``label`` is deliberately plain-language ("Searching your documents") because it is read by
    someone who does not know what an embedding is. ``detail`` carries the specific fact ("routed
    to Devanagari", "8 passages, best 0.62") for the collapsed summary — one payload, two
    audiences.

    Steps are emitted as running/done pairs sharing an ``id`` so the frontend upserts rather than
    appending twice.
    """

    id: str
    node: NodeName
    status: Literal["running", "done"]
    label: str
    detail: str | None = None


@dataclass(frozen=True)
class Done:
    """The final, fully-assembled grounded answer (emitted after the last ``Token``)."""

    answer: GeneratedAnswer


AgentEvent = Token | Step | Done


def emit(event: AgentEvent) -> None:
    """Push an event onto the graph's custom stream channel, if anyone is listening.

    A no-op in two cases, both deliberate: under ``ainvoke`` (the blocking ``/v1/query`` and
    non-streaming chat paths) LangGraph installs a no-op writer, and outside a graph run entirely
    (a node called directly from a unit test) ``get_stream_writer`` raises ``RuntimeError``.
    Swallowing both means nodes have exactly one code path whether or not the caller subscribed —
    which is why ``generate`` can always stream internally and let blocking callers just collect.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)
