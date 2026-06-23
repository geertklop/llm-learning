"""Retrieve relevant medical context from pgvector for a list of symptoms."""

import logging

import ollama
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from agents.config import Settings
from agents.single_agent.state import GuidelineResult

logger = logging.getLogger(__name__)

# Must match the embedding model dimensionality used during RAG ingestion.
# bge-m3 produces 1024-dimensional vectors.
_EMBEDDING_DIMENSIONS = 1024

# Cosine distance ranges from 0 (identical) to 2 (opposite directions).
# Documents with a distance above this threshold are considered irrelevant.
# bge-m3 on Dutch medical text: cardiac articles appear at ~0.36-0.40,
# common symptom articles (koorts, etc.) appear at ~0.40-0.45.
# 0.5 is a practical upper bound that captures both without excessive noise.
_DEFAULT_THRESHOLD = 0.5


class _Base(DeclarativeBase):
    """SQLAlchemy declarative base for read-only ORM access."""


class _Guideline(_Base):
    """
    Read-only ORM view of the guidelines table populated by the RAG project.

    Each row represents the "Wanneer bel je de huisarts" section from one
    Thuisarts.nl patient-situation page, indexed with NHG-based triage criteria.

    Attributes
    ----------
    id
        Auto-incrementing primary key.
    url
        Canonical Thuisarts URL.
    title
        Article title (e.g. "Ik heb buikpijn").
    slug
        First URL path segment identifying the condition (e.g. "buikpijn").
    context
        Full text of the triage section, including urgency sub-headings.
    embedding
        768-dimensional vector produced by the embedding model.
    """

    __tablename__ = "guidelines"
    __table_args__ = (
        Index(
            "guidelines_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 30},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(String)
    urgency_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    context: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Vector] = mapped_column(Vector(_EMBEDDING_DIMENSIONS))


def retrieve_guidelines(
    symptoms: list[str],
    settings: Settings,
    top_k: int = 5,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[GuidelineResult]:
    """
    Retrieve the most relevant medical context passages for the given symptoms.

    Each symptom is embedded and queried independently (multi-query retrieval).
    Results are merged, deduplicated by URL, and filtered by a cosine distance
    threshold before returning the top-K most relevant passages.

    Parameters
    ----------
    symptoms
        List of symptom strings extracted from the patient message.
    settings
        Application settings containing Postgres and Ollama configuration.
    top_k
        Maximum number of context passages to return.
    threshold
        Maximum cosine distance to consider a document relevant. Cosine
        distance ranges from 0 (identical) to 2 (opposite). Documents with
        a distance above this value are discarded.

    Returns
    -------
    List of GuidelineResult ordered by relevance (closest first). May be
    shorter than ``top_k`` if fewer documents pass the relevance threshold.
    """
    if not symptoms:
        return []

    dsn = (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    engine = create_engine(dsn)

    # Deduplicate by slug so all three urgency chunks of the same article
    # (spoed / vandaag / geen-spoed) don't consume multiple top-k slots.
    # We keep the chunk with the lowest (best) distance per condition.
    best_by_slug: dict[str, tuple[_Guideline, float]] = {}

    with Session(engine) as session:
        for symptom in symptoms:
            query_vector = _embed(symptom, settings.ollama_embed_model)
            candidates = _query_by_vector(session, query_vector, threshold, top_k)
            for document, distance in candidates:
                existing = best_by_slug.get(document.slug)
                if existing is None or distance < existing[1]:
                    best_by_slug[document.slug] = (document, distance)

    ranked = sorted(best_by_slug.values(), key=lambda pair: pair[1])[:top_k]

    for rank, (document, distance) in enumerate(ranked, start=1):
        logger.debug(
            "[%d/%d] url=%s distance=%.4f urgency=%s title=%s",
            rank,
            len(ranked),
            document.url,
            distance,
            document.urgency_hint,
            document.title,
        )

    return [
        GuidelineResult(
            url=document.url,
            title=document.title,
            urgency_hint=document.urgency_hint,
            context=document.context,
        )
        for document, _ in ranked
    ]


def _query_by_vector(
    session: Session,
    query_vector: list[float],
    threshold: float,
    top_k: int,
) -> list[tuple[_Guideline, float]]:
    """
    Query the documents table for the nearest neighbours of a single vector.

    Parameters
    ----------
    session
        An open SQLAlchemy session.
    query_vector
        The embedding vector to search against.
    threshold
        Maximum cosine distance to include in results.
    top_k
        Maximum number of rows to return per query.

    Returns
    -------
    List of (document, cosine_distance) pairs ordered by distance ascending.
    """
    distance_expr = _Guideline.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(_Guideline, distance_expr)
        .where(distance_expr <= threshold)
        .order_by(distance_expr)
        .limit(top_k)
    )
    return [
        (document, float(distance)) for document, distance in session.execute(statement)
    ]


def _embed(text: str, model: str) -> list[float]:
    """
    Produce an embedding vector for the given text using Ollama.

    Parameters
    ----------
    text
        The text to embed.
    model
        Ollama model name used for embedding. Must match the model used during
        RAG ingestion, otherwise dimension or semantic mismatches will occur.

    Returns
    -------
    A list of floats representing the embedding vector.
    """
    response = ollama.embed(model=model, input=text)
    return response.embeddings[0]
