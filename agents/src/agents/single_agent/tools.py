"""Medical tools for the single agent."""

from langchain_core.tools import tool


@tool
def describe_symptom(symptom: str) -> str:
    """Look up clinical information about a symptom the patient is experiencing.

    This tool is used to provide the agent with information about symptoms that
    the patient is experiencing, which can help the agent make more informed decisions
    about diagnosis and treatment.

    Parameters
    ----------
    symptom
        The symptom to look up (e.g. "fever", "cough", "headache").
    """
    # In a real implementation, this might query a medical database or use an API.
    descriptions = {
        "fever": "A temporary increase in average body temperature of 98.6°F (37°C).",
        "cough": (
            "A sudden, forceful hacking sound to release air"
            " and clear irritation in the throat or airway."
        ),
        "headache": (
            "Pain in any region of the head, often accompanied"
            " by nausea and sensitivity to light and sound."
        ),
    }
    return descriptions.get(symptom.lower(), "Symptom not recognized.")


def check_drug_interactions(new_drug: str, existing_drugs: list[str]) -> str | None:
    """Check for potential interactions between two drugs.

    This tool is used to ensure that any medications prescribed by the agent do not
    have harmful interactions with each other.

    Parameters
    ----------
    new_drug
        The new drug being considered for prescription (e.g. "aspirin").
    existing_drugs
        A list of drugs the patient is currently taking
        (e.g. ["ibuprofen", "warfarin"]).
    """
    # In a real implementation, this might query a medical database or use an API.
    interactions = {
        ("aspirin", "ibuprofen"): "Increased risk of gastrointestinal bleeding.",
        ("aspirin", "warfarin"): "Increased risk of bleeding.",
        ("ibuprofen", "warfarin"): "Increased risk of bleeding.",
    }

    # loop over existing drugs, sort the pair so the dict key is order-independent
    # always sort the two
    for existing_drug in existing_drugs:
        key = tuple(sorted((new_drug.lower(), existing_drug.lower())))
        if key in interactions:
            return interactions[key]

    return None
