# llm-learning

Learn about LLMs and how to interact with them through RAG/MCP/Agents.

This repo isn't the most beautiful engineered code, since it's meant for learning and experimentation. But there are some coding standards to follow, which are outlined in the [Python instructions](.github/instructions/python.instructions.md).

## Workspace structure

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) — a single lockfile (`uv.lock`) at the root, with independent packages as members.

```
llm-learning/          ← workspace root (shared lockfile + dev tools)
├── rag/               ← RAG pipeline (Ollama + pgvector)
└── agents/            ← Agent systems (LangGraph)
```

Dependencies are resolved together and pinned in one place, so all members stay consistent.

## Running commands

```bash
# Run a command scoped to a specific member (from anywhere in the repo)
uv run --package rag rag --help
uv run --package agents agents --help

# Or cd into the member directory and run directly
cd rag && uv run rag query "Do statins reduce atrial fibrillation?"

# Sync all members at once
uv sync --all-packages
```

## Dev tools

Tools in the root `[dependency-groups]` (mypy, ruff) apply to the whole workspace:

```bash
uv run ruff check .
uv run mypy .
```
