"""Alembic migration environment."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from multilingual_rag.core.config import get_settings
from multilingual_rag.db import models as _models
from multilingual_rag.db.base import Base

_ = _models

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def migration_database_url() -> str:
    """Return a sync SQLAlchemy URL for Alembic."""
    url = get_settings().database_url
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "sqlite+aiosqlite://",
        "sqlite://",
    )


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    context.configure(
        url=migration_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = migration_database_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
