"""one document per (user, checksum) — content-addressed dedup

Revision ID: 0003_user_checksum_unique
Revises: 0002_fk_ondelete_cascade
Create Date: 2026-07-18

Document ids are now derived from (user_id, content checksum), so a user re-uploading identical
content updates in place instead of creating duplicates. This constraint enforces that invariant
at the database level.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_user_checksum_unique"
down_revision: str | None = "0002_fk_ondelete_cascade"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_documents_user_checksum", "documents", ["user_id", "checksum"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_user_checksum", "documents", type_="unique")
