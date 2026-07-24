"""Streaming answer generation over an OpenAI-compatible endpoint.

The blocking generator (``openai_compatible_generator.py``) returns a whole answer; this one
yields it token-by-token for the SSE chat endpoint. Streaming is an *async-edge* concern — the
sync RAG core is untouched — so this uses ``openai.AsyncOpenAI`` (a native async token stream)
rather than bridging the sync client through a thread. Retrieval, which *is* the blocking sync
core, is offloaded with ``asyncio.to_thread`` like the ``/v1/query`` route.

Prompting, language resolution, citation parsing, and the OpenAI-error mapping are all shared
with the blocking generator, so a streamed answer is grounded and cited identically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from fastapi import status
from openai import AsyncOpenAI, OpenAIError

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import ConversationTurn, GeneratedAnswer
from multilingual_rag.generation.citations import answer_citations
from multilingual_rag.generation.contextualize import (
    CONTEXTUALIZE_SYSTEM,
    build_contextualize_prompt,
    clean_standalone_query,
)
from multilingual_rag.generation.language import resolve_answer_language
from multilingual_rag.generation.openai_compatible_generator import (
    build_chat_messages,
    generation_app_error,
)
from multilingual_rag.generation.prompts import SYSTEM_INSTRUCTIONS, build_answer_prompt
from multilingual_rag.retrieval.service import RetrievalService


@dataclass(frozen=True)
class Token:
    """One streamed piece of the answer text."""

    text: str


@dataclass(frozen=True)
class Done:
    """The final, fully-assembled grounded answer (emitted after the last ``Token``)."""

    answer: GeneratedAnswer


StreamEvent = Token | Done


class StreamClient(Protocol):
    """A chat client that streams the assistant's reply as text deltas."""

    def astream_completion(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        """Yield assistant message deltas for a system + history + user exchange."""
        ...

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        """Return a whole (non-streamed) completion — used for the condense/query-rewrite call."""
        ...


class OpenAICompatibleStreamClient:
    """Streaming counterpart of ``OpenAICompatibleChatClient`` (async, ``stream=True``)."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def astream_completion(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=cast(Any, build_chat_messages(system, prompt, history)),
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:  # e.g. a trailing usage-only chunk carries no delta
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def acomplete(self, *, model: str, system: str, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=cast(Any, build_chat_messages(system, prompt, ())),
        )
        return response.choices[0].message.content or ""


class StreamingAnswerGenerator:
    """Generate a grounded answer as a stream of ``Token``s followed by one ``Done``."""

    def __init__(
        self,
        settings: Settings,
        *,
        retrieval_service: RetrievalService,
        client: StreamClient | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.model = settings.generation_model
        if client is not None:
            self.client: StreamClient = client
            return
        api_key = settings.generation_api_key
        if api_key is None:
            raise AppError(
                "GENERATION_API_KEY is required to generate answers.",
                code="missing_generation_api_key",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        self.client = OpenAICompatibleStreamClient(
            api_key.get_secret_value(),
            settings.generation_base_url,
            settings.generation_timeout_seconds,
        )

    async def _contextualize(
        self, history: Sequence[ConversationTurn], question: str
    ) -> str:
        """Rewrite a follow-up into a standalone query for retrieval (identity if no history)."""
        if not history:
            return question
        try:
            raw = await self.client.acomplete(
                model=self.model,
                system=CONTEXTUALIZE_SYSTEM,
                prompt=build_contextualize_prompt(history, question),
            )
        except OpenAIError as exc:
            raise generation_app_error(exc, self.model) from exc
        return clean_standalone_query(raw, fallback=question)

    async def stream(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        preferred_language: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Condense the follow-up, retrieve, stream the answer, then emit the assembled ``Done``.

        Retrieval is scoped to the chat's own documents (``session_id``) when given.
        """
        search_query = await self._contextualize(history, query)
        # Retrieval is the blocking sync core (local bge-m3 embed + Chroma) — offload it.
        context = await asyncio.to_thread(
            self.retrieval_service.retrieve,
            search_query,
            user_id=user_id,
            session_id=session_id,
        )
        if search_query != query:
            # Answer the user's actual wording; retrieval used the rewritten standalone query.
            context = context.model_copy(update={"query": query})
        response_language = resolve_answer_language(
            preferred_language, context.query_language, context.results
        )
        prompt = build_answer_prompt(context, response_language=response_language)

        parts: list[str] = []
        try:
            async for delta in self.client.astream_completion(
                model=self.model, system=SYSTEM_INSTRUCTIONS, prompt=prompt, history=history
            ):
                parts.append(delta)
                yield Token(delta)
        except OpenAIError as exc:
            raise generation_app_error(exc, self.model) from exc

        answer = "".join(parts).strip()
        if not answer:
            raise AppError(
                "The generation endpoint returned an empty answer.",
                code="empty_generation_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
        yield Done(
            GeneratedAnswer(
                answer=answer,
                language=response_language,
                citations=answer_citations(answer, context.results),
            )
        )
