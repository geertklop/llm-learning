"""Generate an answer from retrieved context using an Ollama LLM."""

import ollama

from rag.database.schemas import Document

# The system prompt frames the LLM as a grounded reader rather than a general
# knowledge model. Explicitly forbidding answers outside the provided context
# reduces hallucination — the LLM can't fall back to training data when
# something isn't mentioned.
_SYSTEM_PROMPT = (
    "You are a medical research assistant. Answer the user's question using "
    "only the provided context passages. If the answer is not in the context, "
    "say so clearly rather than speculating."
)


def generate(question: str, documents: list[Document], llm_model: str) -> str:
    """
    Produce a grounded answer by sending retrieved context to the LLM.

    Parameters
    ----------
    question
        The user's natural-language question.
    documents
        Documents returned by the retriever; their context fields are
        injected into the prompt so the LLM can cite them.
    llm_model
        Ollama model name used for generation.

    Returns
    -------
    The LLM's answer as a plain string.
    """
    user_message = _build_user_message(question, documents)

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer: str = response.message.content
    return answer


def _build_user_message(question: str, documents: list[Document]) -> str:
    """
    Assemble a single user message that interleaves context and the question.

    Parameters
    ----------
    question
        The user's natural-language question.
    documents
        Retrieved documents whose context fields will be injected.

    Returns
    -------
    A formatted string ready to send as the user turn of the chat.
    """
    passages = "\n\n---\n\n".join(doc.context for doc in documents)
    message = f"Context:\n\n{passages}\n\nQuestion: {question}"
    return message
