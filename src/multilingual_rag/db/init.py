"""Database initialization helpers for local development and tests."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from multilingual_rag.db import models as _models
from multilingual_rag.db.base import Base

_ = _models


async def create_database_schema(engine: AsyncEngine) -> None:
    """Create all database tables for non-migration test/dev workflows."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
