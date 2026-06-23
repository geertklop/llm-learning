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
    stream_input: dict | Command = {
            "messages": history,
            "symptoms": None,
            "medications": None,
            "urgency": None,
            "findings": None,
            "draft_response": None,
            "retrieved_guidelines": None,
            "clarification_round": None,
            "clarification_question": None,
        }

    output = None
    while True:
        stream = graph.stream_events(stream_input, config, version="v3")
        output = stream.output

        if not stream.interrupts:
            break

        interrupt_value = stream.interrupts[0].value

        if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "clarification":
            question = interrupt_value["question"]
            print(f"\n{question}\n")
            answer = input("U: ").strip()
            stream_input = Command(resume=answer)
        else:
            # Doctor review
            print(f"\n[doctor review]\n{interrupt_value}\n")
            edit = input("Press Enter to approve, or type an edited response: ").strip()
            stream_input = Command(resume=edit if edit else interrupt_value)

    if isinstance(output, dict) and "messages" in output:
        return output["messages"]
    return None
