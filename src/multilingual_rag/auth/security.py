"""Password hashing and JWT helpers."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import status

from multilingual_rag.core.config import Settings
from multilingual_rag.core.errors import AppError

PASSWORD_HASH_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    if len(password) < 8:
        raise AppError(
            "Password must be at least 8 characters.",
            code="weak_password",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a PBKDF2-HMAC-SHA256 hash."""
    try:
        algorithm, iterations_text, salt, expected_digest = password_hash.split("$", maxsplit=3)
        iterations = int(iterations_text)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual_digest, expected_digest)


def create_access_token(settings: Settings, *, subject: str, email: str) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "email": email,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "iat": now,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(settings: Settings, token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        decoded = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise AppError(
            "Invalid or expired access token.",
            code="invalid_access_token",
            status_code=status.HTTP_401_UNAUTHORIZED,
        ) from exc
    if not isinstance(decoded, dict) or not decoded.get("sub"):
        raise AppError(
            "Invalid access token payload.",
            code="invalid_access_token",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    return decoded

