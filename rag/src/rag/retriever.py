"""Retrieve the most relevant documents from pgvector for a given question."""

import logging

import ollama
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.database.schemas import Document

logger = logging.getLogger(__name__)


def retrieve(
    question: str, session: Session, embed_model: str, top_k: int
) -> list[Document]:
    """
    Find the top-K documents whose embeddings are closest to the question.

    Cosine distance (<=>) measures the angle between two vectors. A smaller
    angle means the vectors point in similar directions — i.e. similar meaning.
    The ivfflat index makes this fast without scanning every row.

    Parameters
    ----------
    question
        The natural-language question from the user.
    session
        An open SQLAlchemy session.
    embed_model
        Ollama model name used to embed the question.
    top_k
        Number of documents to return.

    Returns
    -------
    List of Document ORM objects ordered by cosine distance (closest first).
    """
    query_vector = _embed_query(question, embed_model)

    # .cosine_distance() is provided by pgvector-python's SQLAlchemy integration.
    # It maps directly to the <=> operator, so the ivfflat index is still used.
    statement = (
        select(Document)
        .order_by(Document.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    documents = list(session.scalars(statement))

    for rank, document in enumerate(documents, start=1):
        # Log at INFO so retrieved context is visible during normal runs,
        # not just when --debug is active.
        logger.info(
            "[%d/%d] pubid=%s question=%s",
            rank,
            len(documents),
            document.pubid,
            document.question,
        )

    return documents


def _embed_query(question: str, model: str) -> list[float]:
    """
    Produce an embedding vector for the user's question.

    Parameters
    ----------
    question
        The natural-language question to embed.
    model
        Ollama model name used for embedding.

    Returns
    -------
    A list of floats representing the question's embedding vector.
    """
    response = ollama.embed(model=model, input=question)
    # ollama.embed accepts a list of inputs, but we always pass a single
    # question, so there is exactly one embedding in the response.
    vector: list[float] = response.embeddings[0]
    return vector
