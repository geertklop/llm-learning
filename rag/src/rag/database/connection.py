"""SQLAlchemy engine and session factory, and schema creation."""

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from rag.config import Settings
from rag.database.schemas import Base


def get_engine(settings: Settings) -> Any:
    """
    Create a SQLAlchemy engine connected to the pgvector database.

    Parameters
    ----------
    settings
        Application settings containing the Postgres DSN.

    Returns
    -------
    A SQLAlchemy engine instance.
    """
    # SQLAlchemy requires the postgresql+psycopg:// scheme to select the
    # psycopg3 driver explicitly. pydantic's PostgresDsn produces postgresql://.
    dsn = str(settings.postgres_dsn).replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    engine = create_engine(dsn)
    return engine


def get_session(settings: Settings) -> Session:
    """
    Create and return a SQLAlchemy session.

    Parameters
    ----------
    settings
        Application settings containing the Postgres DSN.

    Returns
    -------
    An open SQLAlchemy Session. The caller is responsible for committing
    or rolling back and closing it.
    """
    engine = get_engine(settings)
    session = Session(engine)
    return session


def create_schema(settings: Settings) -> None:
    """
    Create the pgvector extension and all ORM-mapped tables if they don't exist.

    Parameters
    ----------
    settings
        Application settings containing the Postgres DSN.
    """
    engine = get_engine(settings)
    with engine.connect() as connection:
        # The vector extension must exist before SQLAlchemy creates the table
        # with a vector(768) column — DDL order matters here.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
    Base.metadata.create_all(engine)
