"""Medical tools for the single agent."""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_THUISARTS_BASE = "https://www.thuisarts.nl/"
_MAX_ARTICLE_CHARS = 6000


@tool
def lookup_thuisarts_article(url: str) -> str:
    """Fetch the full text of a Thuisarts.nl patient information article.

    Use this before writing a recommendation to get the complete, up-to-date
    guideline text. Only call with URLs that were provided in the context.

    Parameters
    ----------
    url
        Full Thuisarts.nl article URL, e.g.
        ``https://www.thuisarts.nl/pijn-op-borst/ik-heb-pijn-op-borst-wat-kan-zijn``
    """
    if not url.startswith(_THUISARTS_BASE):
        return "Error: only thuisarts.nl URLs are supported."

    try:
        response = httpx.get(
            url,
            headers={"User-Agent": "llm-learning/1.0"},
            follow_redirects=True,
            timeout=10,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Error fetching article: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup.find("main")
    if not article:
        return "Article content not found."

    for tag in article.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    text = article.get_text(separator="\n", strip=True)
    return text[:_MAX_ARTICLE_CHARS]


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
