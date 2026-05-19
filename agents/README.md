# Agents

Single and multi-agent systems using LangGraph, applied to medical AI.

## Running commands

This package is a member of the `llm-learning` uv workspace. Commands can be run from the repo root or from this directory:

```bash
# From repo root
uv run --package agents agents --help

# From this directory
uv run agents --help
```

## What is an Agent?

A regular LLM call is one-shot: question in, answer out. An **agent** runs a loop:

```
user question
  → LLM thinks: "I need more information"
  → tool runs → returns result
  → LLM thinks: "Now I can answer"
  → LLM answers → loop ends
```

The loop stops when the LLM produces a response with no tool call. This pattern is called **ReAct** (Reason + Act).

## How Tools Work

The LLM cannot execute code. Instead it outputs a structured JSON tool call request. The **agent runtime** reads that, executes the function, appends the result as a new message, and feeds the updated conversation back to the LLM.

Tools are plain Python functions decorated with `@tool`. The function name, docstring, and type hints are sent to the LLM as its instruction manual for when and how to call the tool.

## State

State is a list of messages that grows with each step:

```
HumanMessage    → the original question
AIMessage       → LLM response (may contain tool call request)
ToolMessage     → tool result
AIMessage       → LLM final answer
```

Each node reads from state, does work, and appends back to it. The `add_messages` reducer ensures messages are appended rather than overwritten.

## System Prompt

A `SystemMessage` is prepended to the message list on every LLM call inside `create_llm_node`. It is never written back into state — state stays clean with only human/AI/tool messages.

```
LLM sees:  [SystemMessage, HumanMessage, AIMessage, ToolMessage, ...]
State has: [HumanMessage, AIMessage, ToolMessage, ...]
```

This is equivalent to sending a system message once at the start of a chat API conversation. The system message token cost is constant (~50 tokens here) and negligible compared to the growing message history. For large system prompts (2,000+ tokens), history trimming or summarization becomes important — covered in Phase 2.

**Effect on tool use:** the system prompt `"Use your tools to look up symptoms"` causes the agent to call `describe_symptom` even when it could answer from training data. Tool docstrings and system prompt wording directly influence which tools the agent reaches for.

```
START
  │
  ▼
[llm]  ◄──────────────┐
  │                   │
  ├─ tool call? ──► [tools]
  │
  └─ no tool call? ──► END
```

| Component         | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| `llm` node        | Calls the LLM with the current message history              |
| `tools` node      | Executes the requested tool, appends result to messages     |
| `tools_condition` | Inspects last AI message — routes to `tools` or `END`       |
| `MessagesState`   | State schema — a list of messages with an append reducer    |
| `MemorySaver`     | (Phase 2) Checkpoints state per `thread_id` for persistence |

## Learning Phases

| Phase | Concept                   | New primitives                                               | Commit    |
| ----- | ------------------------- | ------------------------------------------------------------ | --------- |
| 1     | Single agent — ReAct loop | `StateGraph`, `MessagesState`, `ToolNode`, `tools_condition` | `0501a54` |
| 2     | Memory & persistence      | `SqliteSaver`, `thread_id`, Textual TUI                      | —         |
| 3     | Multi-agent               | Supervisor pattern, `Command`, subgraphs, handoffs           | —         |
| 4     | Agentic RAG               | RAG as a tool, CRAG pattern, pgvector integration            | —         |

## Running

```bash
cp .env.example .env         # add your config
uv run agents chat           # start interactive session
uv run agents chat --debug   # with full LLM prompt logging
```
