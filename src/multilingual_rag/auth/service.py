"""Authentication service."""

from __future__ import annotations

from multilingual_rag.auth.repository import UserRepository
from multilingual_rag.auth.security import create_access_token, hash_password, verify_password
from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError
from multilingual_rag.core.models import UserRecord


class AuthService:
    """Handle signup and login workflows."""

    def __init__(self, settings: Settings, repository: UserRepository) -> None:
        self.settings = settings
        self.repository = repository

    async def signup(self, *, email: str, password: str) -> tuple[UserRecord, str]:
        """Create a user and access token."""
        user = await self.repository.create_user(email=email, password_hash=hash_password(password))
        token = create_access_token(self.settings, subject=user.user_id, email=user.email)
        return user, token

    async def login(self, *, email: str, password: str) -> tuple[UserRecord, str]:
        """Validate credentials and return a user plus access token."""
        user_row = await self.repository.get_by_email(email)
        if user_row is None or not verify_password(password, user_row.password_hash):
            raise AppError(
                "Invalid email or password.",
                code="invalid_credentials",
                status_code=401,
            )
        user = UserRecord(user_id=user_row.id, email=user_row.email)
        token = create_access_token(self.settings, subject=user.user_id, email=user.email)
        return user, token
