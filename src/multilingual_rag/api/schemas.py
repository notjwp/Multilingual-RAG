"""Shared API response schemas."""

from typing import Any

from pydantic import BaseModel, Field

from multilingual_rag.core.config import Environment


class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    details: dict[str, Any] | None = Field(default=None, description="Optional error details.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str
    environment: Environment


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str = "ok"
    service: str
    environment: Environment
    ready: bool
    checks: dict[str, bool]

