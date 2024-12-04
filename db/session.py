from contextlib import contextmanager
from typing import Generator

from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.settings import db_settings

# Create SQLAlchemy Engine using a database URL
db_url: str = db_settings.get_db_url()
db_engine: Engine = create_engine(db_url, pool_pre_ping=True)

# Create a SessionLocal class
# https://fastapi.tiangolo.com/tutorial/sql-databases/#create-a-sessionlocal-class
SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False, autoflush=False, bind=db_engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session.

    Yields:
        Session: An SQLAlchemy database session.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager to get a database session.

    Yields:
        Session: An SQLAlchemy database session.
    """
    db = next(get_db())  # Get the database session
    try:
        yield db
    finally:
        db.close()
