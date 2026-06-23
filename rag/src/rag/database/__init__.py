"""Database package: ORM models, connection factory, and schema creation."""

from rag.database.connection import create_schema, get_engine, get_session
from rag.database.schemas import (
    EMBEDDING_DIMENSIONS,
    GUIDELINE_EMBEDDING_DIMENSIONS,
    Base,
    Document,
    Guideline,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "GUIDELINE_EMBEDDING_DIMENSIONS",
    "Base",
    "Document",
    "Guideline",
    "create_schema",
    "get_engine",
    "get_session",
]
