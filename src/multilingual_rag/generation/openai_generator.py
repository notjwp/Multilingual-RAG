"""OpenAI answer generation implementation."""

from __future__ import annotations

from typing import Protocol, cast

from fastapi import status
from openai import OpenAI, OpenAIError

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import AnswerCitation, GeneratedAnswer, RetrievalContext
from multilingual_rag.generation.citations import parse_cited_results
from multilingual_rag.generation.prompts import SYSTEM_INSTRUCTIONS, build_answer_prompt


class _ResponseWithText(Protocol):
    @property
    def output_text(self) -> str:
        """Return SDK-concatenated response text."""
        ...


class _OpenAIResponsesClient(Protocol):
    def create_response(self, *, model: str, instructions: str, prompt: str) -> _ResponseWithText:
        """Create a model response."""
        ...


class OpenAIResponsesClient:
    """Small adapter around the OpenAI Responses API."""

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def create_response(self, *, model: str, instructions: str, prompt: str) -> _ResponseWithText:
        """Create a response using the OpenAI SDK."""
        response = self._client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
        return cast(_ResponseWithText, response)


class OpenAIAnswerGenerator:
    """Generate grounded answers with OpenAI."""

    def __init__(self, settings: Settings, *, client: _OpenAIResponsesClient | None = None) -> None:
        api_key = settings.openai_api_key
        if client is None and api_key is None:
            raise AppError(
                "OPENAI_API_KEY is required to generate answers.",
                code="missing_openai_api_key",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self.model = settings.openai_generation_model
        if client is not None:
            self.client = client
        else:
            if api_key is None:
                raise AssertionError("api_key should be set after validation")
            self.client = OpenAIResponsesClient(api_key.get_secret_value())

    def generate_answer(
        self,
        *,
        context: RetrievalContext,
        preferred_language: str | None = None,
    ) -> GeneratedAnswer:
        """Generate a grounded answer from retrieved context."""
        response_language = preferred_language or context.query_language
        prompt = build_answer_prompt(context, response_language=response_language)

        try:
            response = self.client.create_response(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                prompt=prompt,
            )
        except OpenAIError as exc:
            raise AppError(
                "OpenAI answer generation failed.",
                code="openai_generation_error",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

        answer = response.output_text.strip()
        if not answer:
            raise AppError(
                "OpenAI returned an empty answer.",
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

