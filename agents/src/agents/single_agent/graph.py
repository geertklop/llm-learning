"""Single-agent LangGraph graph definition."""

from agents.config import Settings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    NodeNames,
    create_classify_node,
    create_medication_interactions_node,
    create_respond_node,
    create_triage_node,
    doctor_review,
    escalate,
)
from .state import MedicalState


def create_graph(settings: Settings) -> CompiledStateGraph:
    """Create the multi-node medical agent graph.

    Graph shape
    -----------
    ::

        START → classify
        classify → triage           (symptoms present)
        classify → interactions     (medications only, no symptoms)
        classify → respond          (general query)
        triage → escalate           (urgency == "red") → doctor_review → END
        triage → interactions       (urgency != "red" and medications present)
        triage → respond            (urgency != "red", no medications)
        interactions → respond
        respond → doctor_review
        doctor_review → END

    Nodes
    -----
    classify : LLM call with structured output that extracts symptoms and
        medications from the user message. Routes via Command(goto=...).
    triage : LLM call that evaluates symptoms for urgency, sets the
        ``urgency`` field (red/orange/yellow/green), and routes via
        Command(goto=...).
    interactions : Checks extracted medications for known interactions.
        Always routes to ``respond``.
    escalate : Builds an emergency warning from symptoms, stores it in
        ``draft_response``, and routes to ``doctor_review``. No interrupt here —
        the doctor reviews via ``doctor_review``.
    respond : Final LLM call. Reads the full MedicalState (symptoms,
        medications, urgency) to produce a calibrated draft response stored
        in ``draft_response``. Does not append to ``messages``.
    doctor_review : Uses interrupt() to surface the draft to a human doctor
        for approval or editing. Appends the approved response to ``messages``.

    State
    -----
    MedicalState — custom TypedDict with four fields:
    ``messages``, ``symptoms``, ``medications``, ``urgency``.
    Routing decisions live in nodes (Command), not in the state itself.
    """
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host)

    graph = StateGraph(state_schema=MedicalState)
    graph.add_node(NodeNames.CLASSIFY, create_classify_node(llm))
    graph.add_node(NodeNames.TRIAGE, create_triage_node(llm))
    graph.add_node(NodeNames.INTERACTIONS, create_medication_interactions_node(llm))
    graph.add_node(NodeNames.RESPOND, create_respond_node(llm))
    graph.add_node(NodeNames.ESCALATE, escalate)
    graph.add_node(NodeNames.DOCTOR_REVIEW, doctor_review)

    graph.add_edge(START, NodeNames.CLASSIFY)

    return graph.compile(checkpointer=MemorySaver())
