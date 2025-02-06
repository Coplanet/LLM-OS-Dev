"""user sessions

Revision ID: 58e47f509ef9
Revises: f7e9aa4908e2
Create Date: 2025-02-06 09:26:06.580723

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "58e47f509ef9"
down_revision = "f7e9aa4908e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "user_sessions",
        sa.Column("session_id", sa.String(length=50), nullable=False),
        sa.Column("user_id", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("meta", sa.Text(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "user_id", name="_session_user_uc"),
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_sessions_id"),
        "user_sessions",
        ["id"],
        unique=False,
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_sessions_session_id"),
        "user_sessions",
        ["session_id"],
        unique=False,
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_sessions_user_id"),
        "user_sessions",
        ["user_id"],
        unique=False,
        schema="public",
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        op.f("ix_public_user_sessions_user_id"),
        table_name="user_sessions",
        schema="public",
    )
    op.drop_index(
        op.f("ix_public_user_sessions_session_id"),
        table_name="user_sessions",
        schema="public",
    )
    op.drop_index(
        op.f("ix_public_user_sessions_id"),
        table_name="user_sessions",
        schema="public",
    )
    op.drop_table("user_sessions", schema="public")
    # ### end Alembic commands ###
