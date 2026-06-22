"""ORM models and shared metadata for the RAG database."""

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Must match the output dimensionality of the configured embedding model.
# Changing the model without recreating the table causes dimension mismatch errors.
EMBEDDING_DIMENSIONS = 768

# bge-m3 produces 1024-dimensional vectors. Kept separate from EMBEDDING_DIMENSIONS
# so the documents table (nomic-embed-text / 768-dim) is not affected.
GUIDELINE_EMBEDDING_DIMENSIONS = 1024

# bge-m3 produces 1024-dimensional vectors. Kept separate from EMBEDDING_DIMENSIONS
# so the documents table (nomic-embed-text / 768-dim) is not affected.
GUIDELINE_EMBEDDING_DIMENSIONS = 1024


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


class Guideline(Base):
    """
    ORM model representing a single Thuisarts.nl triage article.

    Each row corresponds to the "Wanneer bel je de huisarts" section of one
    patient-situation page (e.g. /buikpijn/ik-heb-buikpijn). The context
    contains the structured urgency criteria (spoed / vandaag / niet spoed)
    exactly as written by NHG-affiliated physicians.

    Attributes
    ----------
    id
        Auto-incrementing primary key.
    url
        Canonical Thuisarts URL with a fragment identifying the urgency
        sub-section (e.g. /buikpijn/ik-heb-buikpijn#spoed). The UNIQUE
        constraint makes re-runs idempotent: existing rows are skipped.
    title
        Article title extracted from the h1 element.
    slug
        First path segment of the URL (e.g. "buikpijn"). Useful for
        grouping all situation pages under the same condition.
    urgency_hint
        Urgency level implied by the sub-section heading: "red" (spoed),
        "orange" (vandaag bellen), "yellow" (wel bellen), or None if the
        heading did not match a recognised pattern.
    context
        Text of the urgency sub-section, prefixed with its heading so the
        embedding captures the urgency signal alongside the criteria.
    embedding
        1024-dimensional vector of title + context, produced by bge-m3
        at index time.
    """

    __tablename__ = "guidelines"
    __table_args__ = (
        Index(
            "guidelines_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            # lists ≈ sqrt(expected row count). Thuisarts has ~500 triage
            # pages, so sqrt(500) ≈ 22. Round up to 30 for some headroom.
            postgresql_with={"lists": 30},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(String, index=True)
    urgency_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    context: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(GUIDELINE_EMBEDDING_DIMENSIONS))
