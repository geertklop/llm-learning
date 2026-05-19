# RAG pipeline

A retrieval-augmented generation pipeline built from scratch on top of Ollama and pgvector.

## Learning goal

The point of this project is to understand what RAG _actually does_ — not to ship production software. Every layer is implemented explicitly so the internals are visible:

- **Embeddings** — `ollama.embed()` converts text to vectors; the norm and dimensions are logged so you can see what the model produces.
- **Storage** — pgvector stores and indexes vectors in PostgreSQL; the ivfflat index with cosine distance is configured by hand so the tradeoffs are clear.
- **Retrieval** — a SQLAlchemy ORM query using `.cosine_distance()` finds the nearest neighbours without raw SQL magic.
- **Generation** — `ollama.chat()` receives the retrieved passages as injected context; the system prompt is written explicitly to minimise hallucination.

Building it this way first means that if you later adopt LangChain or a similar framework, you know what it is replacing:

| This project                              | LangChain equivalent                            |
| ----------------------------------------- | ----------------------------------------------- |
| `ollama.embed()` in `retriever.py`        | `OllamaEmbeddings`                              |
| SQLAlchemy + pgvector ORM query           | `PGVector` vectorstore + `.similarity_search()` |
| `_build_user_message()` + `ollama.chat()` | `ChatOllama` + `PromptTemplate` + `LLMChain`    |
| The retrieve → generate flow in `cli.py`  | `RetrievalQA` or an LCEL chain                  |

## Architecture

```
uv run rag ingest       # embed PubMedQA and store vectors in pgvector
uv run rag query "..."  # embed the question, retrieve top-K docs, generate answer
```

```
question
   │
   ├─ embed (nomic-embed-text via Ollama)
   │
   ├─ retrieve top-K by cosine distance (pgvector / PostgreSQL)
   │
   ├─ inject context into prompt
   │
   └─ generate answer (mistral:7b via Ollama)
```

## Stack

- **Ollama** — runs models natively on macOS (Metal GPU); no Docker needed for inference
- **pgvector/pgvector:pg18** — PostgreSQL with vector extension, running in Docker
- **SQLAlchemy 2.0 + psycopg3** — ORM and database driver
- **pydantic-settings** — typed configuration from `.env`
- **HuggingFace `datasets`** — loads PubMedQA (`qiaojin/PubMedQA`, `pqa_labeled` split)

## Setup

> This package is a member of the `llm-learning` uv workspace. Steps 4 and 5 can also be run from the repo root as `uv run --package rag rag ingest` / `uv run --package rag rag query "..."`.

```bash
# 1. Start the database
docker compose up -d

# 2. Copy and fill in config
cp .env.example .env

# 3. Pull the required Ollama models
ollama pull nomic-embed-text
ollama pull mistral:7b

# 4. Index the dataset (first run takes a few minutes)
uv run rag ingest

# 5. Ask a question
uv run rag query "Do statins reduce the risk of atrial fibrillation?"
```

To override the LLM model without editing `.env`:

```bash
LLM_MODEL=llama3.2:3b uv run rag query "Do statins reduce atrial fibrillation?"
```
