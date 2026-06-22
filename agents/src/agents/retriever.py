"""Retrieve relevant medical context from pgvector for a list of symptoms."""

import logging

import ollama
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from agents.config import Settings

logger = logging.getLogger(__name__)

# Must match the embedding model dimensionality used during RAG ingestion.
_EMBEDDING_DIMENSIONS = 768

# Cosine distance ranges from 0 (identical) to 2 (opposite directions).
# Documents with a distance above this threshold are considered irrelevant.
_DEFAULT_THRESHOLD = 0.4


class _Base(DeclarativeBase):
    """SQLAlchemy declarative base for read-only ORM access."""


class _Document(_Base):
    """
    Read-only ORM view of the documents table populated by the RAG project.

    Attributes
    ----------
    id
        Auto-incrementing primary key.
    pubid
        Unique PubMedQA identifier.
    question
        The research question from PubMedQA.
    context
        Concatenated abstract paragraphs used as retrieved context.
    embedding
        768-dimensional vector produced by the embedding model.
    """

    __tablename__ = "documents"
    __table_args__ = (
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
    embedding: Mapped[Vector] = mapped_column(Vector(_EMBEDDING_DIMENSIONS))


def retrieve_guidelines(
    symptoms: list[str],
    settings: Settings,
    top_k: int = 5,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[str]:
    """
    Retrieve the most relevant medical context passages for the given symptoms.

    Each symptom is embedded and queried independently (multi-query retrieval).
    Results are merged, deduplicated by document ID, and filtered by a cosine
    distance threshold before returning the top-K most relevant passages.

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
    List of context strings ordered by relevance (closest first). May be
    shorter than ``top_k`` if fewer documents pass the relevance threshold.
    """
    if not symptoms:
        return []

    dsn = (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    engine = create_engine(dsn)

    # best_by_pubid maps pubid -> (document, best_distance) across all symptom queries.
    best_by_pubid: dict[str, tuple[_Document, float]] = {}

    with Session(engine) as session:
        for symptom in symptoms:
            query_vector = _embed(symptom, settings.ollama_embed_model)
            candidates = _query_by_vector(session, query_vector, threshold, top_k)
            for document, distance in candidates:
                existing = best_by_pubid.get(document.pubid)
                if existing is None or distance < existing[1]:
                    best_by_pubid[document.pubid] = (document, distance)

    ranked = sorted(best_by_pubid.values(), key=lambda pair: pair[1])[:top_k]

    for rank, (document, distance) in enumerate(ranked, start=1):
        logger.debug(
            "[%d/%d] pubid=%s distance=%.4f question=%s",
            rank,
            len(ranked),
            document.pubid,
            distance,
            document.question,
        )

    return [document.context for document, _ in ranked]


def _query_by_vector(
    session: Session,
    query_vector: list[float],
    threshold: float,
    top_k: int,
) -> list[tuple[_Document, float]]:
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
    distance_expr = _Document.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(_Document, distance_expr)
        .where(distance_expr <= threshold)
        .order_by(distance_expr)
        .limit(top_k)
    )
    return [(document, float(distance)) for document, distance in session.execute(statement)]


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
