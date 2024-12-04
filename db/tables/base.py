from sqlalchemy import Column, DateTime, Integer, MetaData, func, orm
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy model definitions.

    https://fastapi.tiangolo.com/tutorial/sql-databases/#create-a-base-class
    https://docs.sqlalchemy.org/en/20/orm/mapping_api.html#sqlalchemy.orm.DeclarativeBase
    """

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    metadata = MetaData(schema="public")

    def save(self, db: orm.Session):
        db.add(self)
        db.commit()
        db.refresh(self)
        return self
