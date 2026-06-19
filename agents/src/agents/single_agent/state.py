from typing import TypedDict, Literal


from langchain_core.messages import BaseMessage

URGENCY_OPTIONS = Literal["red", "orange", "yellow", "green"]


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
    """

    messages: list[BaseMessage]
    symptoms: list[str] | None
    medications: list[str] | None
    urgency: URGENCY_OPTIONS | None

    findings: list[str] | None
    draft_response: str | None
