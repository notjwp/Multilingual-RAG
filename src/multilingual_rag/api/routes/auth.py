"""Authentication routes."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.auth.dependencies import get_current_user
from multilingual_rag.auth.repository import UserRepository
from multilingual_rag.auth.security import create_access_token
from multilingual_rag.auth.service import AuthService
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import UserRecord
from multilingual_rag.db.session import get_session

router = APIRouter(prefix="/v1/auth", tags=["auth"])
CURRENT_USER_DEPENDENCY = Depends(get_current_user)
SESSION_DEPENDENCY = Depends(get_session)


class AuthRequest(BaseModel):
    """Signup/login request."""

    email: EmailStr
    password: str = Field(min_length=8)


class AuthResponse(BaseModel):
    """Authentication response."""

    access_token: str
    token_type: str = "bearer"
    user: UserRecord


@router.post("/signup", response_model=AuthResponse)
async def signup(
    body: AuthRequest,
    request: Request,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> AuthResponse:
    """Create a new user and return an access token."""
    settings = cast(Settings, request.app.state.settings)
    user, token = await AuthService(settings, UserRepository(session)).signup(
        email=body.email,
        password=body.password,
    )
    await session.commit()
    return AuthResponse(access_token=token, user=user)


@router.post("/login", response_model=AuthResponse)
async def login(
    body: AuthRequest,
    request: Request,
    session: AsyncSession = SESSION_DEPENDENCY,
) -> AuthResponse:
    """Authenticate a user and return an access token."""
    settings = cast(Settings, request.app.state.settings)
    user, token = await AuthService(settings, UserRepository(session)).login(
        email=body.email,
        password=body.password,
    )
    return AuthResponse(access_token=token, user=user)


@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    request: Request,
    current_user: UserRecord = CURRENT_USER_DEPENDENCY,
) -> AuthResponse:
    """Issue a fresh access token for the current user (sliding session; needs a valid token)."""
    settings = cast(Settings, request.app.state.settings)
    token = create_access_token(settings, subject=current_user.user_id, email=current_user.email)
    return AuthResponse(access_token=token, user=current_user)


@router.get("/me", response_model=UserRecord)
async def me(current_user: UserRecord = CURRENT_USER_DEPENDENCY) -> UserRecord:
    """Return the current authenticated user."""
    return current_user
