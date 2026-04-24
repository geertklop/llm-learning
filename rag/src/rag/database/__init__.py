"""Database package: ORM models, connection factory, and schema creation."""

from rag.database.connection import create_schema, get_engine, get_session
from rag.database.schemas import Base, Document, EMBEDDING_DIMENSIONS

__all__ = [
    "Base",
    "Document",
    "EMBEDDING_DIMENSIONS",
    "create_schema",
    "get_engine",
    "get_session",
]
