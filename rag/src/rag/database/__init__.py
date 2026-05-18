"""Database package: ORM models, connection factory, and schema creation."""

from rag.database.connection import create_schema, get_engine, get_session
from rag.database.schemas import EMBEDDING_DIMENSIONS, Base, Document

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Base",
    "Document",
    "create_schema",
    "get_engine",
    "get_session",
]
