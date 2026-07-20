"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from multilingual_rag import __version__
from multilingual_rag.api.routes.auth import router as auth_router
from multilingual_rag.api.routes.chat import router as chat_router
from multilingual_rag.api.routes.documents import jobs_router
from multilingual_rag.api.routes.documents import router as documents_router
from multilingual_rag.api.routes.health import router as health_router
from multilingual_rag.api.routes.query import router as query_router
from multilingual_rag.api.schemas import ErrorResponse
from multilingual_rag.core.config import Settings, get_settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(app_settings.log_level)
        yield

    app = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(query_router)
    app.include_router(chat_router)
    register_exception_handlers(app)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register application-wide exception handlers."""

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        response = ErrorResponse(error=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=response.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        response = ErrorResponse(
            error="validation_error",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response.model_dump(),
        )


app = create_app()
