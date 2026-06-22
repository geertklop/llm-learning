"""Command-line interface for the RAG pipeline."""

import argparse
import logging

from datasets import load_dataset

from rag.config import Settings
from rag.database import create_schema, get_session
from rag.generator import generate
from rag.indexer import ingest
from rag.retriever import retrieve
from rag.thuisarts_indexer import ingest_guidelines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Medical RAG pipeline using Ollama and pgvector.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Initialise the database and index documents."
    )
    ingest_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging (shows embedding dimensions, norm, sample values).",
    )

    guidelines_parser = subparsers.add_parser(
        "ingest-guidelines",
        help="Scrape Thuisarts.nl triage articles and index them into the guidelines table.",
    )
    guidelines_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging (shows skipped pages and embedding details).",
    )

    query_parser = subparsers.add_parser(
        "query", help="Ask a question and get a grounded answer."
    )
    query_parser.add_argument("question", help="The medical question to answer.")
    query_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable INFO/DEBUG logging (shows retrieved documents and embeddings).",
    )

    return parser


def main() -> None:
    """Entry point for the rag CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    settings = Settings()

    if args.command == "ingest":
        if args.debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s"
            )
        create_schema(settings)
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        session = get_session(settings)
        ingest(dataset, session, settings.embed_model)
        session.close()

    if args.command == "ingest-guidelines":
        if args.debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s"
            )
        else:
            logging.basicConfig(
                level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
            )
        create_schema(settings)
        session = get_session(settings)
        ingest_guidelines(session, settings.guidelines_embed_model)
        session.close()

    if args.command == "query":
        if args.debug:
            logging.basicConfig(
                level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
            )
        session = get_session(settings)
        documents = retrieve(
            question=args.question,
            session=session,
            embed_model=settings.embed_model,
            top_k=settings.top_k,
        )
        session.close()

        print(f"\nRetrieved {len(documents)} documents.\n")
        answer = generate(
            question=args.question,
            documents=documents,
            llm_model=settings.llm_model,
        )
        print(answer)

        print("\nSources:")
        for rank, document in enumerate(documents, start=1):
            print(f"  [{rank}] pubid={document.pubid} — {document.question}")


if __name__ == "__main__":
    main()
