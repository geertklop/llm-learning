"""Index PubMedQA documents into pgvector."""

import ollama
from datasets import Dataset
from sqlalchemy.orm import Session

from rag.database.crud import insert_document


def _join_context(paragraphs: list[str]) -> str:
    """
    Join a list of abstract paragraphs into a single context string.

    Parameters
    ----------
    paragraphs
        Individual abstract paragraphs from a PubMedQA record.

    Returns
    -------
    A single string with paragraphs separated by newlines.
    """
    joined = "\n".join(paragraphs)
    return joined


def _embed(text: str, model: str) -> list[float]:
    """
    Produce a single embedding vector for a piece of text.

    Parameters
    ----------
    text
        The text to embed.
    model
        Ollama model name to use for embedding.

    Returns
    -------
    A list of floats representing the embedding vector.
    """
    response = ollama.embed(model=model, input=text)
    vector = response.embeddings[0]
    return vector


def ingest(dataset: Dataset, session: Session, embed_model: str) -> None:
    """
    Embed each record in the dataset and upsert it into pgvector.

    Each record is embedded by concatenating its question and context.
    The embedding captures the semantic meaning of both, so at query time
    a question-only embedding can still retrieve the most relevant records.

    Parameters
    ----------
    dataset
        A HuggingFace Dataset where each row has ``pubid``, ``question``,
        and ``context.contexts`` fields.
    session
        An open SQLAlchemy session. Closed by the caller after this
        function returns.
    embed_model
        Ollama model name used to produce embedding vectors.
    """
    total = len(dataset)

    for index, record in enumerate(dataset):
        pubid = str(record["pubid"])
        question = record["question"]
        context = _join_context(record["context"]["contexts"])

        # Embed question + context together so the vector encodes both the
        # query intent and the supporting evidence in a single representation.
        text_to_embed = f"{question}\n\n{context}"
        embedding = _embed(text_to_embed, embed_model)

        insert_document(
            session=session,
            pubid=pubid,
            question=question,
            context=context,
            embedding=embedding,
        )
        session.commit()

        print(f"[{index + 1}/{total}] indexed pubid={pubid}")

    print("Indexing complete.")
