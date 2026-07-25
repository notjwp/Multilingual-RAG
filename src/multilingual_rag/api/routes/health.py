"""Health and readiness routes."""

from typing import cast

from fastapi import APIRouter, Request

from multilingual_rag.api.schemas import HealthResponse, ReadinessResponse
from multilingual_rag.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    """Return process health for load balancers and uptime checks."""
    settings = _get_settings(request)
    return HealthResponse(service=settings.app_name, environment=settings.environment)


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(request: Request) -> ReadinessResponse:
    """Return readiness status for dependency-aware startup checks."""
    settings = _get_settings(request)
    # Only require the OpenAI key when OpenAI embeddings are actually selected. The default stack
    # (local bge-m3 + an OpenAI-compatible generation endpoint such as NIM) never calls OpenAI, so
    # gating readiness on that key reported ready=false forever in a correct free deployment.
    checks = {
        "configuration": True,
        "embeddings_configured": (
            settings.openai_api_key is not None
            if settings.embedding_provider == "openai"
            else True
        ),
    }
    return ReadinessResponse(
        service=settings.app_name,
        environment=settings.environment,
        ready=all(checks.values()),
        checks=checks,
    )


def _get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)
