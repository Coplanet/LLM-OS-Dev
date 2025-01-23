from typing import Optional

from sqlalchemy import Column, DateTime, Integer, MetaData, delete, func, orm
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

    def save(self, db: Optional[orm.Session] = None, auto_commit: bool = True):
        def _save(db_: orm.Session) -> None:
            db_.add(self)
            if auto_commit or not db:
                db_.commit()
                db_.refresh(self)

        if db:
            _save(db)
        else:
            from db.session import get_db_context

            with get_db_context() as db:
                _save(db)

        return self

    def delete(
        self, db: Optional[orm.Session] = None, auto_commit: bool = True
    ) -> None:
        def _delete(db: orm.Session) -> None:
            db.execute(delete(self.__class__).where(self.__class__.id == self.id))
            if auto_commit or not db:
                db.commit()

        if db:
            _delete(db)
        else:
            from db.session import get_db_context

            with get_db_context() as db:
                _delete(db)
