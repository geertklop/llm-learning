"""Command-line interface for the RAG pipeline."""

import argparse

from datasets import load_dataset

from rag.config import Settings
from rag.database import create_schema, get_session
from rag.indexer import ingest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Medical RAG pipeline using Ollama and pgvector.",
    )
    parser.add_argument(
        "--model",
        help="Override the LLM model from settings (e.g. gemma3:4b).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Initialise the database and index documents.")

    return parser


def main() -> None:
    """Entry point for the rag CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    settings = Settings()
    if args.model:
        settings.llm_model = args.model

    if args.command == "ingest":
        create_schema(settings)
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        session = get_session(settings)
        ingest(dataset, session, settings.embed_model)
        session.close()


if __name__ == "__main__":
    main()
