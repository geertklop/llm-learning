"""Single-agent LangGraph graph definition."""

from functools import partial

from agents.config import Settings
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import check_drug_interactions, describe_symptom

_LLM_NODE = "llm"
_TOOLS_NODE = "tools"
_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a medical information assistant. "
        "Use your tools to look up symptoms and drug interactions. "
        "Always recommend consulting a qualified healthcare professional "
        "before making any medical decisions."
    )
)


def create_llm_node(llm: ChatOllama, state: MessagesState) -> dict:
    """Invoke the LLM with the system prompt prepended to the current state."""
    response = llm.invoke([_SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


def create_graph(settings: Settings) -> CompiledStateGraph:
    """Create the Agent Graph."""
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host)
    tools = [describe_symptom, check_drug_interactions]
    llm_with_tools = llm.bind_tools(tools)

    llm_node = partial(create_llm_node, llm_with_tools)

    graph = StateGraph(state_schema=MessagesState)
    graph.add_node(_LLM_NODE, llm_node)
    graph.add_node(_TOOLS_NODE, ToolNode(tools))

    graph.add_edge(START, _LLM_NODE)
    graph.add_conditional_edges(_LLM_NODE, tools_condition)
    graph.add_edge(_TOOLS_NODE, _LLM_NODE)

    # tool_condition automatically adds END to the graph whenever there
    # was no tool call in the LLM response.

    return graph.compile()
