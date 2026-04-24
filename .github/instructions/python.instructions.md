---
name: "Python Standards"
description: "Coding conventions for Python files"
applyTo: "**/*.py"
---

## Software engineering principles

Follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) to create maintainable code with clear separation of concerns.

Apply GRASP principles (especially Information Expert, Controller, and Low Coupling) to place behavior where data and responsibility naturally belong.

Apply KISS (Keep It Simple Stupid) and DRY (Don't Repeat Yourself) principles — favor simplicity over over-engineering and extract common logic to avoid duplication.

Follow YAGNI (You Aren't Gonna Need It) — don't implement features speculatively; implement only what's needed now.

Use The Zen of Python as a tiebreaker for design choices: explicit is better than implicit, simple is better than complex, and readability counts.

## Readability and maintainability

Always prioritize readability and clarity.

Avoid abbreviations in function and variable names unless the abbreviation is universally understood (e.g. `url`, `id`, `http`). Prefer `parse_dependency_name` over `parse_dep_name`, `configuration` over `config` where context is ambiguous.

Handle edge cases and write clear exception handling.

Keep cyclomatic complexity low: split complex branches into helper functions and target Ruff McCabe `C901` compliance (max complexity is 5 in this repo).

Prefer existing dependencies in this repo or Python standard library modules before introducing new packages.

Don't do multi-line returns — assign to a variable and return that variable instead. This allows for easier debugging and readability.

## Docstrings

Use numpy-style docstrings without type hints. Example:

```python
def embed_documents(texts: list[str], model: str) -> list[list[float]]:
    """
    Generate vector embeddings for a list of text documents.

    Parameters
    ----------
    texts
        Input documents to embed.
    model
        Name of the embedding model to use.

    Returns
    -------
    List of embedding vectors, one per input document.
    """
```

Note that type hints in functions are always necessary, but they should not be included in docstrings.

Always limit the length of every line of docstrings to 88 characters. This is important because our linters cannot automatically format docstrings, so we must ensure they are properly formatted from the start.

## Comments

Write comments to explain _why_ code exists, not _what_ it does. Comment non-obvious design decisions or complex logic.

```python
# Preferred: explains reasoning
# Truncate to model max tokens to avoid silent context loss at inference time
tokens = tokens[:max_length]

# Avoid: restates what code obviously does
timestamp = datetime.now()  # Set timestamp to now
```

For algorithm-related code, include explanations of the approach used.

## Strings

Use f-strings for all string formatting — they're more readable than format() or %.

```python
# Preferred
message = f"Retrieved {len(chunks)} chunks for query: {query}"

# Avoid
message = "Retrieved {} chunks for query: {}".format(len(chunks), query)
```

## Testing

When writing tests, use pytest.

Name test files with a `test_` prefix (e.g., `test_retriever.py`) placed alongside or under a `tests/` folder mirroring the source structure.

Prefer testing with real objects and data over mocks. When mocking is necessary, use `unittest.mock` — never the `mocker` fixture.

## Functions

Avoid nested functions when possible, as they are hard to test.

Keep functions short and focused on a single task as much as possible.

## Code blocks

Use `pass` only when a code block genuinely requires no operation (rare in practice).

Prefer explicit None checks for None default arguments instead of mutable defaults — this prevents unexpected shared state.

```python
# Preferred
def process_records(data, cache=None):
    if cache is None:
        cache = {}

# Avoid
def process_records(data, cache={}):
    pass
```

Avoid exception silencing — if catching an exception, handle it properly or re-raise it with context.

```python
# Preferred: handle or provide context
try:
    response = client.embed(text)
except httpx.HTTPStatusError as error:
    logger.warning(f"Embedding request failed: {error.response.status_code}")
    raise

# Avoid: silently swallowing errors
try:
    response = client.embed(text)
except:
    pass
```

Flatten deeply nested code structures to improve readability — use early returns, guards, and helper methods.

```python
# Preferred: guard clause eliminates nesting
def validate_document(document: Document) -> bool:
    if not document.text:
        return False
    if len(document.text) > MAX_CHUNK_SIZE:
        return False
    return is_supported_language(document)

# Avoid: deeply nested conditions
if document.text:
    if len(document.text) <= MAX_CHUNK_SIZE:
        if is_supported_language(document):
            return True
```

For unimplemented abstract methods use ellipsis (...); for abstract concrete methods use raise NotImplementedError.

```python
# Abstract base class method
def abstract_method(self):
    ...

# Concrete class method not yet implemented
def future_method(self):
    raise NotImplementedError("Coming in next sprint")
```

## Constants and Magic Values

Do not use magic values directly in code.

Promote a value to a module-level constant only when reused across multiple functions, classes, or module boundaries.

If a value is local to a single function, keep it local with a descriptive variable name. Add a comment if the value's meaning is not obvious, or to explain why a certain value is given.

Use leading underscore for module-private constants.

```python
# At top of file — reused across multiple functions
MAX_CHUNK_SIZE = 512
_DEFAULT_TEMPERATURE = 0.7

# Local value inside a function when not reused elsewhere
overlap_ratio = 0.1  # 10% overlap between chunks reduces context loss at boundaries
chunk_overlap = int(chunk_size * overlap_ratio)
```

## Tooling and workflow

When adding Python dependencies, update `pyproject.toml` using uv (`uv add <package>` from the relevant subpackage directory).

## Imports

Always import libraries at the top of the file and never inside functions.

## File Naming

- For file names, adhere to PEP8 and abbreviate when appropriate. We prefer single-word file names, but if multiple words are necessary, separate them with underscores and use snake case. For example, use `data_loader.py` instead of `dataLoader.py` or `DataLoader.py`.

- When naming files after classes, use snake_case of the class name (for example, `DocumentRetriever` -> `document_retriever.py`).

## Source of truth

Treat this instruction file as the source of truth for coding guidance in this repository.

## Python environments

Each subpackage in this monorepo manages its own `.venv`. Always activate the `.venv` local to the subpackage you are working in. Do not use global Python environments. If no `.venv` exists for a subpackage, inform the user.
