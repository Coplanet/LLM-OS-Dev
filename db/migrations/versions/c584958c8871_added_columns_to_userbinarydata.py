"""added columns to UserBinaryData

Revision ID: c584958c8871
Revises: fe6de506b130
Create Date: 2025-01-08 13:44:22.227871

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c584958c8871"
down_revision = "fe6de506b130"
branch_labels = None
depends_on = None


def upgrade():
    # Add columns to UserBinaryData table
    op.add_column(
        'user_binary_data',
        sa.Column('data_compressed', sa.LargeBinary(), nullable=True)
    )
    op.add_column(
        'user_binary_data',
        sa.Column('data_hashsum', sa.String(255), nullable=True)
    )
    op.create_index(
        'ix_user_binary_data_data_hashsum', 'user_binary_data', ['data_hashsum']
    )
    op.add_column(
        'user_binary_data',
        sa.Column('data_compressed_hashsum', sa.String(255), nullable=True)
    )
    op.create_index(
        'ix_user_binary_data_data_compressed_hashsum', 'user_binary_data', ['data_compressed_hashsum']
    )

def downgrade():
    # Remove columns and indexes from UserBinaryData table
    op.drop_index('ix_user_binary_data_data_compressed_hashsum', table_name='user_binary_data')
    op.drop_column('user_binary_data', 'data_compressed_hashsum')
    op.drop_index('ix_user_binary_data_data_hashsum', table_name='user_binary_data')
    op.drop_column('user_binary_data', 'data_hashsum')
    op.drop_column('user_binary_data', 'data_compressed')
