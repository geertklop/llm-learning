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
        description="All medications mentioned in the message. Empty list if none.",
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
