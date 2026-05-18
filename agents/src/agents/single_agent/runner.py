"""Streaming multi-turn interactive chat runner for the medical agent."""

from agents.config import Settings
from agents.single_agent.graph import create_graph
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph


def run_chat() -> None:
    """
    Run an interactive multi-turn chat loop with the single agent.

    Maintains message history from the final graph state after each turn.
    Type 'exit' or 'quit' to end the session.
    """
    settings = Settings()
    graph = create_graph(settings)
    history: list[BaseMessage] = []

    print("Medical assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        history.append(HumanMessage(content=question))
        updated = _run_turn(graph, history)
        if updated:
            history = updated


def _run_turn(
    graph: CompiledStateGraph, history: list[BaseMessage]
) -> list[BaseMessage] | None:
    """Stream one conversation turn and return the updated message history.

    stream_mode=["values", "messages"] yields two event types:
      ("messages", (chunk, metadata)) — token-by-token LLM output
      ("values", state)               — full state snapshot after each node
    We use "messages" for live printing and "values" to update history.
    The system prompt is prepended inside create_llm_node and never stored
    in state, so history slicing remains clean.
    """
    final_state = None
    in_ai_response = False

    for event in graph.stream(
        {"messages": history}, stream_mode=["values", "messages"]
    ):
        mode, data = event

        if mode == "messages":
            chunk, metadata = data
            node = metadata.get("langgraph_node", "")

            if node == "tools" and chunk.content:
                # Tool results arrive complete, not streamed.
                print(f"\ntool: {chunk.content}\n")
                in_ai_response = False

            elif node == "llm" and chunk.content:
                # Stream LLM text token by token.
                # Chunks without content are tool-call JSON — skip them.
                if not in_ai_response:
                    print("ai: ", end="", flush=True)
                    in_ai_response = True
                print(chunk.content, end="", flush=True)

        elif mode == "values":
            final_state = data

    if in_ai_response:
        print("\n")

    if final_state:
        return final_state["messages"]
    return None
