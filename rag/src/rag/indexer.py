"""Index PubMedQA documents into pgvector."""

import logging
import math

import ollama
from datasets import Dataset
from sqlalchemy.orm import Session

from rag.database.crud import insert_document

logger = logging.getLogger(__name__)


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
    # ollama.embed accepts a list of inputs for batch efficiency, but we pass a
    # single string, so there is always exactly one embedding in the response.
    vector = response.embeddings[0]

    # L2 norm measures the vector's magnitude. nomic-embed-text returns
    # normalised vectors, so this should be very close to 1.0. A value
    # far from 1.0 would indicate the model is not producing unit vectors,
    # which matters because cosine similarity assumes unit normalisation.
    norm = math.sqrt(sum(value * value for value in vector))
    logger.debug(
        "embedding produced: dimensions=%d, norm=%.6f, sample=%s",
        len(vector),
        norm,
        [round(value, 4) for value in vector[:5]],
    )

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
