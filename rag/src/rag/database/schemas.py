"""ORM models and shared metadata for the RAG database."""

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Must match the output dimensionality of the configured embedding model.
# Changing the model without recreating the table causes dimension mismatch errors.
EMBEDDING_DIMENSIONS = 768


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""


class Document(Base):
    """
    ORM model representing a single indexed PubMed abstract.

    Attributes
    ----------
    id
        Auto-incrementing primary key.
    pubid
        Unique PubMedQA identifier. The UNIQUE constraint makes ingest
        idempotent: re-running it skips rows that already exist.
    question
        The research question from PubMedQA.
    context
        Concatenated abstract paragraphs — the text shown to the LLM.
    embedding
        768-dimensional vector produced by the embedding model. pgvector
        stores and indexes this for similarity search.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # ivfflat index with cosine distance. Cosine measures the angle between
        # vectors rather than magnitude, which is appropriate for text embeddings
        # because scaling a vector doesn't change the meaning of the text.
        Index(
            "documents_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    pubid: Mapped[str] = mapped_column(String, unique=True)
    question: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
