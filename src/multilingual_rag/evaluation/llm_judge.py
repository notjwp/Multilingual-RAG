"""LLM-as-judge faithfulness scoring over any OpenAI-compatible endpoint.

Implements the ``FaithfulnessJudge`` protocol from ``evaluation/faithfulness.py``, reusing the
same chat client (and therefore the same ``GENERATION_*`` config) as answer generation. Opt-in
only: it costs one model call per example against a rate-limited free tier, so callers sample.
"""

from __future__ import annotations

from fastapi import status
from openai import OpenAIError

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.generation.openai_compatible_generator import (
    ChatClient,
    OpenAICompatibleChatClient,
)

JUDGE_SYSTEM = (
    "You are a strict grader for retrieval-augmented answers. Decide whether EVERY factual "
    "claim in the answer is directly supported by the provided context. Reply with exactly one "
    "word: SUPPORTED or UNSUPPORTED. If any claim is missing from the context, reply "
    "UNSUPPORTED."
)


def _judge_prompt(answer: str, context: str) -> str:
    return f"Context:\n{context}\n\nAnswer:\n{answer}\n\nIs every claim supported?"


class LlmFaithfulnessJudge:
    """Judge answer faithfulness with the configured generation model."""

    def __init__(self, settings: Settings, *, client: ChatClient | None = None) -> None:
        api_key = settings.generation_api_key
        if client is None and api_key is None:
            raise AppError(
                "GENERATION_API_KEY is required to judge faithfulness.",
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

    def is_supported(self, *, answer: str, context: str) -> bool:
        """Return True when the judge says every claim is grounded in the context."""
        try:
            verdict = self.client.create_completion(
                model=self.model,
                system=JUDGE_SYSTEM,
                prompt=_judge_prompt(answer, context),
            )
        except OpenAIError as exc:
            raise AppError(
                "Faithfulness judging failed.",
                code="faithfulness_judge_error",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

        # Grade conservatively: anything that isn't a clear SUPPORTED counts as unsupported.
        return verdict.strip().upper().startswith("SUPPORTED")
