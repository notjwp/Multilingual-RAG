"""chat FK ondelete cascade + citation text, so chat sessions can be deleted

Revision ID: 0004_chat_fk_cascade
Revises: 0003_user_checksum_unique
Create Date: 2026-07-20

The chat tables (milestone 14) created their FKs with no ON DELETE rule, so deleting a chat that
had messages — or a message that had citations — would raise a ForeignKeyViolationError (the same
defect Phase D fixed for documents). Recreate the chat FKs with ON DELETE CASCADE, and add the
cited-snippet ``text`` column to message_citations.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_chat_fk_cascade"
down_revision: str | None = "0003_user_checksum_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (child table, fk column, parent table, parent column)
_FKS = (
    ("chat_sessions", "user_id", "users", "id"),
    ("messages", "session_id", "chat_sessions", "id"),
    ("message_citations", "message_id", "messages", "id"),
)


def upgrade() -> None:
    for table, column, parent, parent_col in _FKS:
        name = f"{table}_{column}_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(name, table, parent, [column], [parent_col], ondelete="CASCADE")
    op.add_column("message_citations", sa.Column("text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("message_citations", "text")
    for table, column, parent, parent_col in _FKS:
        name = f"{table}_{column}_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(name, table, parent, [column], [parent_col])
