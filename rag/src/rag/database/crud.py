"""Create, read, update, and delete operations for the documents table."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rag.database.schemas import Document, Guideline


def insert_document(
    session: Session,
    pubid: str,
    question: str,
    context: str,
    embedding: list[float],
) -> None:
    """
    Insert a document row, skipping it silently if the pubid already exists.

    Parameters
    ----------
    session
        An open SQLAlchemy session. The caller is responsible for
        committing after one or more calls to this function.
    pubid
        Unique PubMedQA identifier used as the conflict target.
    question
        The research question text.
    context
        Concatenated abstract paragraphs to pass to the LLM as context.
    embedding
        768-dimensional vector produced by the embedding model.
    """
    statement = (
        insert(Document)
        .values(
            pubid=pubid,
            question=question,
            context=context,
            embedding=embedding,
        )
        # Re-running ingest must be safe: skip rows that already exist rather
        # than raising an error or overwriting data.
        .on_conflict_do_nothing(index_elements=["pubid"])
    )
    session.execute(statement)


def insert_guideline(
    session: Session,
    url: str,
    title: str,
    slug: str,
    urgency_hint: str | None,
    context: str,
    embedding: list[float],
) -> None:
    """
    Insert a guideline row, skipping it silently if the URL already exists.

    Parameters
    ----------
    session
        An open SQLAlchemy session. The caller is responsible for committing.
    url
        Canonical Thuisarts URL with urgency fragment, used as conflict target.
    title
        Article title (h1 text).
    slug
        First URL path segment identifying the condition (e.g. "buikpijn").
    urgency_hint
        Urgency level of this chunk: "red", "orange", "yellow", or None.
    context
        Text of the urgency sub-section prefixed with its heading.
    embedding
        768-dimensional vector produced by the embedding model.
    """
    statement = (
        insert(Guideline)
        .values(
            url=url,
            title=title,
            slug=slug,
            urgency_hint=urgency_hint,
            context=context,
            embedding=embedding,
        )
        .on_conflict_do_nothing(index_elements=["url"])
    )
    session.execute(statement)
