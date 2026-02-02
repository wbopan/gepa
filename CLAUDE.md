# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEPA (Genetic-Pareto) is a Python framework for optimizing text components of systems (AI prompts, code snippets, textual specifications) using LLM-based reflection and evolutionary search. It achieves optimization through iterative mutation, reflection, and Pareto-aware candidate selection.

## Development Commands

```bash
# Setup (using uv - recommended)
uv sync --all-extras --python 3.11

# Run all tests
uv run pytest tests/

# Linting and formatting (via pre-commit)
uv run pre-commit install    # one-time setup
uv run pre-commit run        # check staged files
```

## Code Style

- Google Python Style Guide
- Ruff for linting/formatting (line length: 120)
- Relative imports banned (except in tests)
- Type checking with Pyright (standard mode)
- All comments must be written in English

## Development Guidelines

When modifying existing code or adding new features:

- **Breaking changes allowed**: Feel free to make breaking changes without considering backward compatibility.
- **Minimize changes**: Make the smallest possible modification that achieves the goal. Avoid refactoring unrelated code or adding unnecessary abstractions.
- **Follow existing patterns**: Study how similar functionality is implemented elsewhere in the codebase and mirror those conventions.
- **Logging**: Use `get_logger()` from `gepa.logging` instead of Python's built-in `logging` or `print()`. Use `logger.log(msg, header="...")` for important messages with semantic headers (e.g., `"select"`, `"score"`, `"error"`), and `logger.debug()` for verbose output (enabled via `LOG_LEVEL=DEBUG`).
- **Experiment tracking**: Runs are logged to Weights & Biases (wandb). Use the MCP wandb tools to query run metrics and check experiment status (entity: `bmpixel`, project: `gepa-boost`).
