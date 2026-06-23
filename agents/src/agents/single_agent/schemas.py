"""Pydantic schemas for structured LLM output in the medical agent graph."""

from pydantic import BaseModel, Field

from .state import URGENCY_OPTIONS


class MessageClassification(BaseModel):
    """Schema for classifying a user message into symptoms and medications."""

    symptoms: list[str] = Field(
        default_factory=list,
        description="All medical symptoms or complaints mentioned in the message. Empty list if none.",
    )
    medications: list[str] = Field(
        default_factory=list,
        description="Medications the patient explicitly states they are currently taking or have taken. Do NOT include medications the patient is merely asking about or considering.",
    )


class TriageOutput(BaseModel):
    """Schema for triaging based on symptoms into an urgency level."""

    urgency: URGENCY_OPTIONS = Field(
        description=(
            "The urgency level determined by the triage node based on the symptoms. "
            'One of "red" (emergency, escalate immediately), "orange" (urgent, advise same-day care), '
            '"yellow" (non-urgent, book appointment), or "green" (routine, no action needed).'
        )
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "Set to True only if one focused follow-up question would significantly "
            "change the urgency assessment. Do not ask if urgency is already clear."
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "A single concise question in the patient's own language. "
            "Only set when needs_clarification is True."
        ),
    )
