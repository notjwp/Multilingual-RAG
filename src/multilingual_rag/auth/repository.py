"""User repository backed by SQLAlchemy."""

from __future__ import annotations

from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import UserRecord
from multilingual_rag.db.models import User


class UserRepository:
    """Persist and fetch users."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_user(self, *, email: str, password_hash: str) -> UserRecord:
        """Create a user with a unique normalized email."""
        normalized_email = normalize_email(email)
        existing_user = await self.get_by_email(normalized_email, required=False)
        if existing_user is not None:
            raise AppError(
                "A user with this email already exists.",
                code="email_already_registered",
                status_code=status.HTTP_409_CONFLICT,
            )

        user = User(email=normalized_email, password_hash=password_hash)
        self.session.add(user)
        await self.session.flush()
        return UserRecord(user_id=user.id, email=user.email)

    async def get_by_email(self, email: str, *, required: bool = True) -> User | None:
        """Return a user ORM row by email."""
        result = await self.session.execute(
            select(User).where(User.email == normalize_email(email))
        )
        user = result.scalar_one_or_none()
        if user is None and required:
            raise AppError(
                "Invalid email or password.",
                code="invalid_credentials",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        return user

    async def get_record_by_id(self, user_id: str) -> UserRecord:
        """Return a user record by ID."""
        user = await self.session.get(User, user_id)
        if user is None:
            raise AppError(
                "User not found.",
                code="user_not_found",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        return UserRecord(user_id=user.id, email=user.email)


def normalize_email(email: str) -> str:
    """Normalize an email address for identity comparisons."""
    return email.strip().lower()
