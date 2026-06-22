from typing import TypedDict, Literal


from langchain_core.messages import BaseMessage

URGENCY_OPTIONS = Literal["red", "orange", "yellow", "green"]


class GuidelineResult(TypedDict):
    """
    A single retrieved guideline chunk stored in the graph state.

    Attributes
    ----------
    url
        Canonical Thuisarts URL with urgency fragment (e.g. #spoed-bel-direct).
    title
        Article title (e.g. "Ik heb buikpijn").
    urgency_hint
        Urgency level implied by the sub-section: "red", "orange", "yellow",
        or None if unrecognised.
    context
        Text of the urgency sub-section used as LLM context.
    """

    url: str
    title: str
    urgency_hint: str | None
    context: str


class MedicalState(TypedDict):
    """State to be used accross the medical answering and triaging process

    Parameters
    ----------
    messages
        The accumulated message history from the conversation so far.
    symptoms
        The list of symptoms extracted from the user's input by the classify node.
    medications
        The list of medications extracted from the user's input by the classify node.
    urgency
        The urgency level determined by the triage node based on the symptoms.
    findings
        Any additional findings or warnings generated during the graph execution,
        such as drug interaction warnings.
    draft_response
        The AI-generated draft response produced by the respond node. Preserved
        after doctor review for audit; the doctor-approved text is appended to
        ``messages`` instead.
    retrieved_guidelines
        Medical guideline chunks retrieved from pgvector by the triage node,
        used to ground the urgency assessment. Each entry includes the source
        URL, title, urgency hint, and context text.
    """

    messages: list[BaseMessage]
    symptoms: list[str] | None
    medications: list[str] | None
    urgency: URGENCY_OPTIONS | None
    findings: list[str] | None
    draft_response: str | None
    retrieved_guidelines: list[GuidelineResult] | None
