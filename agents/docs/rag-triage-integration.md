# RAG Integration Plan: Grounding Triage in Medical Guidelines

## Goal

Replace the pure LLM triage step with a retrieval-augmented approach.
The `triage` node will query a vector database of medical urgency guidelines
before calling the LLM, grounding the urgency classification in authoritative
source material rather than LLM training memory.

---

## What Changes

| Layer | Change |
|---|---|
| `state.py` | Add `retrieved_guidelines: list[str] \| None` for audit trail |
| `nodes.py` / `triage` | Retrieve guidelines before LLM call; inject into prompt and state |
| `agents/src/agents/retriever.py` | New module — vector DB query function |
| `agents/src/agents/indexer.py` | New module — indexes guideline documents |
| Data | NHG / Thuisarts urgency criteria, chunked per symptom cluster |
| `pyproject.toml` | Add pgvector / psycopg dependencies (mirror from `rag`) |

Graph shape does **not** change — retrieval stays inside the `triage` node.

---

## Implementation Steps

### 1. Copy the retriever pattern from `rag`

Study `rag/src/rag/retriever.py`:
- Database connection (psycopg + pgvector)
- Embedding model (nomic-embed-text via Ollama)
- Vector similarity query

Create `agents/src/agents/retriever.py` with a single function:

```python
def retrieve_guidelines(symptoms: list[str], top_k: int = 5) -> list[str]:
    """Query the vector DB with symptom text, return top-k guideline chunks."""
```

### 2. Index medical urgency guidelines

Create `agents/src/agents/indexer.py` with a small indexing script.

Source material options:
- NHG urgency criteria (Dutch GP guidelines)
- Thuisarts.nl symptom urgency descriptions
- Static JSON/CSV of symptom → urgency mappings as a starting point

Each document chunk should be one coherent guideline rule, e.g.:
> "Chest pain with shortness of breath: RED — call emergency services immediately."

### 3. Update `create_triage_node`

Before the LLM call:

```python
guidelines = retrieve_guidelines(state["symptoms"])
# Add to state for audit
# Inject into the triage prompt as context
```

The structured output call stays the same — just richer context in the prompt.

### 4. Update `MedicalState`

```python
retrieved_guidelines: list[str] | None
# Guidelines retrieved from vector DB during triage; preserved for audit.
```

---

## Design Decisions

- **Retrieval inside `triage` node** — no new graph node. Keeps graph shape stable.
- **Separate projects** — agents does not import from rag. Retriever logic is duplicated
  intentionally as a learning exercise.
- **Same pgvector DB** — both projects can share the same Postgres instance
  (different tables/collections).
- **Embeddings** — reuse `nomic-embed-text` via Ollama (already running locally).

---

## Out of Scope (for now)

- Streaming guideline sources to the user
- Caching retrieved guidelines across turns
- Fallback when the DB is unavailable (degrade to pure LLM triage)
