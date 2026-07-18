"""Answer generation against any OpenAI-compatible chat-completions endpoint.

The provider is a URL, not a code path. NVIDIA NIM, OpenRouter, Groq, a local Ollama/vLLM shim,
and OpenAI itself all speak the same ``chat.completions`` API, so one adapter serves all of them
— switching is a ``GENERATION_BASE_URL`` change with no code edit.

Everything grounding-related is shared with the rest of the pipeline: prompts, citation parsing,
and answer-language resolution. Model catalogs rotate, so a vanished model surfaces as an
actionable error naming ``GENERATION_MODEL`` rather than a mystery 502.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from fastapi import status
from openai import APITimeoutError, NotFoundError, OpenAI, OpenAIError, RateLimitError

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import AnswerCitation, GeneratedAnswer, RetrievalContext
from multilingual_rag.generation.citations import parse_cited_results
from multilingual_rag.generation.language import resolve_answer_language
from multilingual_rag.generation.prompts import SYSTEM_INSTRUCTIONS, build_answer_prompt


class ChatClient(Protocol):
    def create_completion(self, *, model: str, system: str, prompt: str) -> str:
        """Return the assistant's message text for one system+user exchange."""
        ...


class OpenAICompatibleChatClient:
    """Thin wrapper over the OpenAI SDK pointed at any compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def create_completion(self, *, model: str, system: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=cast(
                Any,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            ),
        )
        return response.choices[0].message.content or ""


class OpenAICompatibleAnswerGenerator:
    """Generate grounded answers. Satisfies the ``AnswerGenerator`` protocol."""

    def __init__(self, settings: Settings, *, client: ChatClient | None = None) -> None:
        api_key = settings.generation_api_key
        if client is None and api_key is None:
            raise AppError(
                "GENERATION_API_KEY is required to generate answers.",
                code="missing_generation_api_key",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self.model = settings.generation_model
        if client is not None:
            self.client: ChatClient = client
        else:
            if api_key is None:
                raise AssertionError("api_key should be set after validation")
            self.client = OpenAICompatibleChatClient(
                api_key.get_secret_value(),
                settings.generation_base_url,
                settings.generation_timeout_seconds,
            )

    def generate_answer(
        self,
        *,
        context: RetrievalContext,
        preferred_language: str | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer grounded in retrieved context."""
        response_language = resolve_answer_language(
            preferred_language, context.query_language, context.results
        )
        prompt = build_answer_prompt(context, response_language=response_language)

        try:
            answer = self.client.create_completion(
                model=self.model,
                system=SYSTEM_INSTRUCTIONS,
                prompt=prompt,
            ).strip()
        except RateLimitError as exc:
            raise AppError(
                "Generation rate limit reached; retry shortly.",
                code="generation_rate_limited",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            ) from exc
        except APITimeoutError as exc:
            raise AppError(
                f"Generation model '{self.model}' did not respond in time. Free-tier models "
                "can be cold or overloaded — try GENERATION_MODEL=meta/llama-3.1-8b-instruct, "
                "or raise GENERATION_TIMEOUT_SECONDS.",
                code="generation_timeout",
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            ) from exc
        except NotFoundError as exc:
            raise AppError(
                f"Generation model '{self.model}' is unavailable at this endpoint. Model "
                "catalogs rotate: pick a current model id from your provider and set "
                "GENERATION_MODEL (or point GENERATION_BASE_URL elsewhere).",
                code="generation_model_unavailable",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc
        except OpenAIError as exc:
            raise AppError(
                "Answer generation failed.",
                code="generation_error",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

        if not answer:
            raise AppError(
                "The generation endpoint returned an empty answer.",
                code="empty_generation_response",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        return GeneratedAnswer(
            answer=answer,
            language=response_language,
            citations=tuple(
                AnswerCitation(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    source=result.source,
                    page=result.page,
                    text=result.text,
                )
                for result in parse_cited_results(answer, context.results)
            ),
        )
