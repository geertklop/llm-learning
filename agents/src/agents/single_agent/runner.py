"""Streaming multi-turn interactive chat runner for the medical agent."""

import logging
from uuid import uuid4

from agents.config import Settings
from agents.single_agent.graph import create_graph
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

logger = logging.getLogger(__name__)


def run_chat() -> None:
    """Run an interactive multi-turn chat loop with the medical agent.

    Maintains message history from the final graph state after each turn.
    Each session gets a unique thread_id so the MemorySaver checkpointer
    can persist state across interrupt/resume cycles.
    Type 'exit' or 'quit' to end the session.
    """
    settings = Settings()
    graph = create_graph(settings)
    config = {"configurable": {"thread_id": str(uuid4())}}
    history: list[BaseMessage] = []

    print("Medical assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        history.append(HumanMessage(content=question))
        updated = _run_turn(graph, history, config)
        if updated:
            history = updated


def _run_turn(
    graph: CompiledStateGraph,
    history: list[BaseMessage],
    config: dict,
) -> list[BaseMessage] | None:
    """Run a single turn: invoke the graph, handle doctor_review interrupt.

    Parameters
    ----------
    graph
        The compiled state graph.
    history
        The accumulated message history from the conversation so far.
    config
        LangGraph run config, must include ``thread_id`` for checkpointing.
    """
    stream = graph.stream_events(
        {
            "messages": history,
            "symptoms": None,
            "medications": None,
            "urgency": None,
            "findings": None,
            "draft_response": None,
            "retrieved_guidelines": None,
        },
        config,
        version="v3",
    )
    _ = stream.output  # drives graph to completion or pause at interrupt

    if stream.interrupts:
        draft = stream.interrupts[0].value
        print(f"\n[doctor review]\n{draft}\n")
        edit = input("Press Enter to approve, or type an edited response: ").strip()
        approved = edit if edit else draft

        resumed = graph.stream_events(Command(resume=approved), config, version="v3")
        output = resumed.output
    else:
        output = stream.output

    if isinstance(output, dict) and "messages" in output:
        return output["messages"]
    return None
