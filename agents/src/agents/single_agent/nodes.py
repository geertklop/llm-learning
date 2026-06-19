"""Node functions for the medical agent graph.

Each function corresponds to one node in the graph defined in graph.py.
Nodes communicate via Command(update={...}, goto="...") for routing and
state updates, or interrupt() for human-in-the-loop pauses.
"""

from enum import StrEnum
import itertools
from typing import Callable

from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.types import Command, interrupt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .tools import check_drug_interactions

from .state import MedicalState, URGENCY_OPTIONS
from .schemas import MessageClassification, TriageOutput


class NodeNames(StrEnum):
    """Enum of node names in the medical agent graph, used for routing."""

    CLASSIFY = "classify"
    TRIAGE = "triage"
    INTERACTIONS = "interactions"
    RESPOND = "respond"
    ESCALATE = "escalate"
    DOCTOR_REVIEW = "doctor_review"


def create_classify_node(llm: ChatOllama) -> Callable[..., Command]:
    """Create the classify node function."""
    classifier = llm.with_structured_output(MessageClassification)

    def classify(state: MedicalState) -> Command:
        """LLM node that classifies the user message into symptoms and medications.

        Expects the latest user message to be the last item in state["messages"].
        Uses structured output parsing to extract symptoms and medications into
        separate fields in the state for downstream nodes to use.
        """

        classification_prompt = SystemMessage(
            content=(
                "You are a medical assistant. Extract any mentioned "
                "symptoms and medications from the user's message."
            )
        )

        result = classifier.invoke([classification_prompt, *state["messages"]])

        if result.symptoms:
            goto = NodeNames.TRIAGE
        # Only do interactions if there are multiple medications called.
        elif len(result.medications) > 1:
            goto = NodeNames.INTERACTIONS
        else:
            goto = NodeNames.RESPOND

        return Command(
            # Put the classified symptoms and medications into the state for downstream
            # nodes to use it.
            update={
                "symptoms": result.symptoms,
                "medications": result.medications,
            },
            goto=goto,
        )

    return classify


def create_triage_node(llm: ChatOllama) -> Callable[..., Command]:
    """Create the triage node function."""
    triagist = llm.with_structured_output(TriageOutput)

    def triage(state: MedicalState) -> Command:
        """LLM node that evaluates the urgency of the user's symptoms.

        Expects symptoms to be present in the state. Uses structured output parsing
        to classify the urgency level as "red", "orange", "yellow", or "green".
        Routes to different nodes based on the urgency and presence of medications.
        """
        symptoms_formatted = "\n- ".join(state["symptoms"])

        triage_prompt = HumanMessage(
            content=(
                "You are a medical assistant. Based on the following symptoms, "
                "classify the urgency level as 'red' (emergency), 'orange' (urgent, advise same-day care), "
                "'yellow' (non-urgent, book appointment), or 'green' (routine, no action needed)."
                f"\n- {symptoms_formatted}"
            )
        )

        result = triagist.invoke([triage_prompt])

        if result.urgency == "red":
            goto = NodeNames.ESCALATE
        elif state["medications"]:
            goto = NodeNames.INTERACTIONS
        else:
            goto = NodeNames.RESPOND

        return Command(update={"urgency": result.urgency}, goto=goto)

    return triage


def create_medication_interactions_node(llm: ChatOllama) -> Callable[..., Command]:
    """Create the medication interactions node function."""

    def interactions(state: MedicalState) -> Command:
        """Node that checks for drug interactions based on the medications in the state.

        This is a placeholder implementation. In a real system, this would likely
        call an external drug interaction API or database instead of using the LLM.
        For demonstration, it just checks if "medication_a" and "medication_b" are
        both present and flags a potential interaction.
        """
        meds = state["medications"]

        # loop over all combinations of medications and check for interactions
        # first, create a matrix for all options
        interactions = []
        for med1, med2 in itertools.combinations(meds, 2):
            interaction = check_drug_interactions(med1, [med2])
            if interaction:
                interactions.append(interaction)

        if interactions:
            return Command(update={"findings": interactions}, goto=NodeNames.RESPOND)

        return Command(goto=NodeNames.RESPOND)

    return interactions


def create_respond_node(llm: ChatOllama) -> Callable[..., Command]:
    """Create the respond node function."""

    def respond(state: MedicalState) -> Command:
        """LLM node that drafts the final response using all available state context.

        Reads symptoms, medications, urgency, and findings from state to produce
        a calibrated response. Stores the result in ``draft_response`` without
        appending to ``messages`` — the approved text is added by ``doctor_review``.
        """
        urgency = state["urgency"]
        symptoms = state["symptoms"] or []
        medications = state["medications"] or []
        findings = state["findings"] or []

        context_lines = [
            "You are a medical information assistant. Draft a response to the patient.",
            f"Urgency level: {urgency or 'unknown'}",
        ]
        if symptoms:
            context_lines.append("Symptoms reported: " + ", ".join(symptoms))
        if medications:
            context_lines.append("Medications mentioned: " + ", ".join(medications))
        if findings:
            context_lines.append("Drug interaction findings:")
            context_lines.extend(f"- {finding}" for finding in findings)

        urgency_instructions = {
            "red": "This is an emergency. Do NOT draft a response — this should have been escalated.",
            "orange": "Advise the patient to seek same-day medical care urgently.",
            "yellow": "Advise the patient to book a non-urgent appointment within a few days.",
            "green": "Provide routine informational guidance. No immediate action needed.",
        }
        if urgency in urgency_instructions:
            context_lines.append(urgency_instructions[urgency])

        system_prompt = SystemMessage(content="\n".join(context_lines))
        result = llm.invoke([system_prompt, *state["messages"]])

        content = result.content
        if isinstance(content, str):
            draft = content
        elif isinstance(content, list):
            draft = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            draft = str(content)
        return Command(update={"draft_response": draft}, goto=NodeNames.DOCTOR_REVIEW)

    return respond


def escalate(state: MedicalState) -> Command:
    """Emergency escalation node for red-urgency situations.

    Builds an emergency warning from the patient's symptoms and stores it as
    ``draft_response`` for the doctor to review before it is sent.
    """
    symptoms = state["symptoms"] or []
    warning = (
        "EMERGENCY: Red-level urgency detected. "
        "Patient requires immediate medical attention.\n"
        "Symptoms: " + ", ".join(symptoms)
    )
    return Command(update={"draft_response": warning}, goto=NodeNames.DOCTOR_REVIEW)


def doctor_review(state: MedicalState) -> Command:
    """Human-in-the-loop node that surfaces the draft response for doctor approval.

    Pauses graph execution via interrupt() to allow a doctor to review and
    optionally edit the AI-generated draft. The approved text is appended to
    ``messages``; ``draft_response`` is preserved as the original AI draft.
    """
    draft = state["draft_response"] or ""
    approved_text = interrupt(draft)
    final_text = approved_text if isinstance(approved_text, str) else draft
    return Command(
        update={"messages": [AIMessage(content=final_text)]},
        goto=END,
    )

