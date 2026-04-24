"""Create, read, update, and delete operations for the documents table."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rag.database.schemas import Document


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
