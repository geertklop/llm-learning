"""Command-line interface for the agents project."""

import argparse

from langchain_core.globals import set_debug

from agents.single_agent.runner import run_chat


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agents",
        description="Single and multi-agent systems using LangGraph.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive multi-turn chat session.",
    )
    chat_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the full prompt and tool schemas sent to the LLM.",
    )

    return parser


def main() -> None:
    """Entry point for the agents CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "chat":
        if args.debug:
            set_debug(True)
        run_chat()


if __name__ == "__main__":
    main()
