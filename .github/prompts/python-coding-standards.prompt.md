---
description: "Align the active Python file with the project coding standards. Use when reviewing or fixing type hints, docstrings, string quoting, and explicitness in a Python file."
agent: "agent"
---

Update the currently active Python file to comply with the Python coding standards defined in [python.instructions.md](./../instructions/python.instructions.md).

Read the instructions file first, then apply all required changes to the active Python file based on this checklist:

- All function and method signatures have complete, specific type hints
- Every function and method has a numpy-style docstring (no type info in docstrings), limited to 88 characters per line
- All strings use double quotes; f-strings used for all formatting
- No implicit truthiness checks on non-boolean types
- Variables and parameters use clear, descriptive names (no unexplained abbreviations)
- No magic values — promote reused literals to named constants
- No mutable default arguments
- No silenced exceptions
- No imports inside functions
