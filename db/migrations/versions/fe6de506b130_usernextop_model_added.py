"""UserNextOp model added

Revision ID: fe6de506b130
Revises: 1253d7b49c63
Create Date: 2025-01-08 02:44:39.641463

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "fe6de506b130"
down_revision = "1253d7b49c63"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "user_next_ops",
        sa.Column("session_id", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_next_ops_id"),
        "user_next_ops",
        ["id"],
        unique=False,
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_next_ops_key"),
        "user_next_ops",
        ["key"],
        unique=False,
        schema="public",
    )
    op.create_index(
        op.f("ix_public_user_next_ops_session_id"),
        "user_next_ops",
        ["session_id"],
        unique=False,
        schema="public",
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        op.f("ix_public_user_next_ops_session_id"),
        table_name="user_next_ops",
        schema="public",
    )
    op.drop_index(
        op.f("ix_public_user_next_ops_key"),
        table_name="user_next_ops",
        schema="public",
    )
    op.drop_index(
        op.f("ix_public_user_next_ops_id"),
        table_name="user_next_ops",
        schema="public",
    )
    op.drop_table("user_next_ops", schema="public")
    # ### end Alembic commands ###
