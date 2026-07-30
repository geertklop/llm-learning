"""Node functions for the medical agent graph.

Each function corresponds to one node in the graph defined in graph.py.
Nodes communicate via Command(update={...}, goto="...") for routing and
state updates, or interrupt() for human-in-the-loop pauses.
"""

import itertools
from collections.abc import Callable
from enum import StrEnum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.types import Command, interrupt

from ..config import Settings
from ..retriever import retrieve_guidelines
from .schemas import MessageClassification, TriageOutput
from .state import MedicalState
from .tools import check_drug_interactions, lookup_thuisarts_article


class NodeNames(StrEnum):
    """Enum of node names in the medical agent graph, used for routing."""

    CLASSIFY = "classify"
    TRIAGE = "triage"
    INTERACTIONS = "interactions"
    RESPOND = "respond"
    CLARIFY = "clarify"
    DOCTOR_REVIEW = "doctor_review"


# Maximum number of patient clarification rounds before forcing a conclusion.
MAX_CLARIFICATION_ROUNDS = 2


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
                "You are a medical assistant. Extract ALL symptoms and medications "
                "mentioned anywhere in the entire conversation — not just the most "
                "recent message. Be specific about anatomical locations to avoid "
                "ambiguity. Only list medications the patient explicitly states they "
                "are currently taking — do NOT extract medications they are asking "
                "about or considering."
            )
        )

        result = classifier.invoke([classification_prompt, *state["messages"]])

        # Merge with any symptoms already in state (e.g. from a previous classify
        # pass before a clarification round) so nothing is dropped.
        prior_symptoms = state.get("symptoms") or []
        prior_medications = state.get("medications") or []
        merged_symptoms = list(dict.fromkeys(prior_symptoms + (result.symptoms or [])))
        merged_medications = list(dict.fromkeys(prior_medications + (result.medications or [])))

        if merged_symptoms:
            goto = NodeNames.TRIAGE
        elif len(merged_medications) > 1:
            goto = NodeNames.INTERACTIONS
        else:
            goto = NodeNames.RESPOND

        return Command(
            update={
                "symptoms": merged_symptoms,
                "medications": merged_medications,
            },
            goto=goto,
        )

    return classify


def create_triage_node(llm: ChatOllama, settings: Settings) -> Callable[..., Command]:
    """Create the triage node function."""
    triagist = llm.with_structured_output(TriageOutput)

    def triage(state: MedicalState) -> Command:
        """LLM node that evaluates the urgency of the user's symptoms.

        Retrieves relevant medical guidelines from pgvector before calling the LLM,
        grounding the urgency assessment in evidence-based context. Uses structured
        output parsing to classify urgency as "red", "orange", "yellow", or "green".
        Routes to different nodes based on the urgency and presence of medications.
        """
        symptoms = state["symptoms"] or []
        # Include the original patient message as an extra query so the full
        # sentence context disambiguates semantically ambiguous symptom terms
        # (e.g. Dutch "borst" = chest OR breast).  The retriever deduplicates
        # by slug, so any overlap with symptom queries is harmless.
        original_message = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            None,
        )
        # Expand lay symptoms to Dutch medical terms to bridge the semantic gap
        # between patient language and the medical terminology used in indexed
        # Thuisarts guidelines (e.g. "pijn op de borst" → "hartkramp").
        if symptoms:
            expansion = llm.invoke([
                SystemMessage(content=(
                    "You are a Dutch GP. Convert the following patient symptoms "
                    "into 3-5 Dutch-language medical condition names as used in "
                    "Dutch patient information (thuisarts.nl style). "
                    "Use everyday Dutch medical terms such as 'hartkramp', "
                    "'blaasontsteking', 'maagontsteking' — NOT Latin terminology. "
                    "Return only the Dutch terms, one per line, no explanations."
                )),
                HumanMessage(content=", ".join(symptoms)),
            ])
            raw = expansion.content if isinstance(expansion.content, str) else ""
            medical_terms = [t.strip("•\u2022-1234567890. ") for t in raw.splitlines() if t.strip()]
        else:
            medical_terms = []
        query_texts = [*symptoms, *medical_terms]
        if original_message:
            query_texts.append(original_message)
        guidelines = retrieve_guidelines(query_texts, settings)

        symptoms_formatted = "\n- ".join(symptoms)
        guidelines_section = (
            "\n\nRelevant medical guidelines:\n"
            + "\n---\n".join(g["context"] for g in guidelines)
            if guidelines
            else ""
        )

        clarification_round = state.get("clarification_round") or 0
        may_ask = clarification_round < MAX_CLARIFICATION_ROUNDS
        clarification_instruction = (
            " If the symptoms are ambiguous or too vague to determine urgency "
            "confidently, set needs_clarification=True and ask one focused question. "
            "This is especially important when you suspect RED (emergency) urgency: "
            "confirm the suspicion with a single targeted question before concluding "
            "it is an emergency — for example asking about pain character, radiation, "
            "or duration. Do not ask for clarification if the symptoms are already "
            "clearly severe."
            if may_ask
            else " Do not ask for clarification — assess based on available information."
        )

        triage_prompt = HumanMessage(
            content=(
                "You are a medical assistant. Based on the following symptoms, "
                "classify the urgency level as 'red' (emergency), 'orange' (urgent, advise same-day care), "
                "'yellow' (non-urgent, book appointment), or 'green' (routine, no action needed)."
                f"{clarification_instruction}"
                f"\n- {symptoms_formatted}"
                f"{guidelines_section}"
            )
        )
        result = triagist.invoke([*state["messages"], triage_prompt])
        if result.needs_clarification and clarification_round < MAX_CLARIFICATION_ROUNDS:
            goto = NodeNames.CLARIFY
        elif not state["medications"] or result.urgency == "red":
            goto = NodeNames.RESPOND
        else:
            goto = NodeNames.INTERACTIONS

        return Command(
            update={
                "urgency": result.urgency,
                "retrieved_guidelines": guidelines,
                "clarification_question": result.clarification_question or None,
            },
            goto=goto,
        )

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
    llm_with_tools = llm.bind_tools([lookup_thuisarts_article])

    def respond(state: MedicalState) -> Command:
        """LLM node that drafts the final response using all available state context.

        Can call ``lookup_thuisarts_article`` to fetch the full current text of
        any Thuisarts.nl guideline before writing the recommendation. Stores the
        result in ``draft_response`` without appending to ``messages`` — the
        approved text is added by ``doctor_review``.
        """
        urgency = state["urgency"]
        symptoms = state["symptoms"] or []
        medications = state["medications"] or []
        findings = state["findings"] or []
        guidelines = state["retrieved_guidelines"] or []

        context_lines = [
            "You are a medical information assistant. Draft a response to the patient "
            "in the same language the patient used.",
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
            "red": "This is a medical emergency. Tell the patient to call 112 immediately or go to the nearest emergency room. Be clear and urgent.",
            "orange": "Advise the patient to seek same-day medical care urgently.",
            "yellow": "Advise the patient to book a non-urgent appointment within a few days.",
            "green": "Provide routine informational guidance. No immediate action needed.",
        }
        if urgency in urgency_instructions:
            context_lines.append(urgency_instructions[urgency])

        if guidelines:
            urls_list = "\n".join(f"- {g['title']}: {g['url']}" for g in guidelines)
            context_lines.append(
                "The following Thuisarts.nl articles were found relevant to the patient's symptoms. "
                "Call lookup_thuisarts_article with the URL to read the full guideline before "
                "drafting your recommendation:\n" + urls_list
            )

        system_prompt = SystemMessage(content="\n".join(context_lines))
        messages: list = [system_prompt, *state["messages"]]

        # Track which articles were actually fetched so only those appear as sources.
        fetched: dict[str, str] = {}  # url → title

        while True:
            result = llm_with_tools.invoke(messages)
            messages.append(result)
            tool_calls = getattr(result, "tool_calls", None) or []
            if not tool_calls:
                break
            for tc in tool_calls:
                url = tc.get("args", {}).get("url", "")
                display = url.removeprefix("https://www.")
                print(f"\n[Tool] lookup_thuisarts_article → {display}")
                tool_output = lookup_thuisarts_article.invoke(tc)
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
                # Match fetched URL back to a title from the retrieved guidelines.
                title = next(
                    (g["title"] for g in guidelines if g["url"] == url),
                    display,
                )
                fetched[url] = title

        content = result.content
        if isinstance(content, str):
            draft = content
        elif isinstance(content, list):
            draft = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            draft = str(content)

        if fetched:
            sources = "\n".join(f"- {title}: {url}" for url, title in fetched.items())
            draft = f"{draft}\n\nBronnen:\n{sources}"

        return Command(update={"draft_response": draft}, goto=NodeNames.DOCTOR_REVIEW)

    return respond


def clarify(state: MedicalState) -> Command:
    """Patient clarification node.

    Pauses graph execution via interrupt() to ask the patient a focused
    follow-up question generated by the triage node. The patient's answer is
    appended to ``messages`` and the graph routes back to ``classify`` so
    symptoms are re-extracted from the enriched conversation before the next
    triage pass.
    """
    question = state.get("clarification_question") or "Kunt u uw klachten nader omschrijven?"
    patient_answer = interrupt({"type": "clarification", "question": question})
    return Command(
        update={
            "messages": [HumanMessage(content=str(patient_answer))],
            "clarification_round": (state.get("clarification_round") or 0) + 1,
        },
        goto=NodeNames.CLASSIFY,
    )


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
