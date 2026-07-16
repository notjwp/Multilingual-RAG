"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import cast

from fastapi import Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.auth.repository import UserRepository
from multilingual_rag.auth.security import decode_access_token
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import UserRecord
from multilingual_rag.db.session import get_session

bearer_scheme = HTTPBearer(auto_error=False)
BEARER_DEPENDENCY = Depends(bearer_scheme)
SESSION_DEPENDENCY = Depends(get_session)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = BEARER_DEPENDENCY,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> UserRecord:
    """Return the authenticated user for a request."""
    injected_user = getattr(request.app.state, "current_user", None)
    if injected_user is not None:
        return cast(UserRecord, injected_user)
    if credentials is None:
        raise AppError(
            "Authentication is required.",
            code="authentication_required",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    settings = cast(Settings, request.app.state.settings)
    payload = decode_access_token(settings, credentials.credentials)
    return await UserRepository(session).get_record_by_id(str(payload["sub"]))
