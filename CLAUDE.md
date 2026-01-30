# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEPA (Genetic-Pareto) is a Python framework for optimizing text components of systems (AI prompts, code snippets, textual specifications) using LLM-based reflection and evolutionary search. It achieves optimization through iterative mutation, reflection, and Pareto-aware candidate selection.

## Development Commands

```bash
# Setup (using uv - recommended)
uv sync --extra dev --python 3.11

# Run all tests
uv run pytest tests/

# Linting and formatting (via pre-commit)
uv run pre-commit install    # one-time setup
uv run pre-commit run        # check staged files
uv run pre-commit run --files path/to/file.py  # check specific files
```

## Architecture

### Core Design Pattern: Protocol-Based Pluggability

GEPA uses Python `Protocol` classes for structural subtyping throughout, enabling duck-typing without inheritance coupling.

### Key Components

**GEPAAdapter** (`core/adapter.py`) - The single integration point for any system:
- `evaluate()`: Execute candidate on batch, return scores and optional trajectories
- `make_reflective_dataset()`: Extract textual information from trajectories for reflection
- `propose_new_texts`: Optional custom proposal logic

**GEPAEngine** (`core/engine.py`) - Main orchestration:
- Runs optimization loop
- Manages Pareto frontier across iterations
- Handles callbacks and state persistence

**GEPAState** (`core/state.py`) - Persistent optimization state:
- Tracks candidates, scores, trajectories
- Supports multiple frontier types: `instance`, `objective`, `hybrid`, `cartesian`
- Enables resumption from checkpoints

**Main API** (`api.py` - `optimize()` function):
- Single entry point with 80+ configurable parameters
- Returns `GEPAResult` with best candidate and metrics

### Proposer Strategies (`proposer/`)

- **ReflectiveMutationProposer**: Uses reflection LM to propose mutations based on failures
- **MergeProposer**: Combines two Pareto-frontier candidates

### Pluggable Strategies (`strategies/`)

| Strategy Type | Options |
|--------------|---------|
| Candidate Selection | `ParetoCandidateSelector`, `CurrentBestCandidateSelector`, `EpsilonGreedyCandidateSelector` |
| Component Selection | `RoundRobinReflectionComponentSelector`, `AllReflectionComponentSelector` |
| Batch Sampling | `EpochShuffledBatchSampler` (default), custom via `BatchSampler` protocol |
| Stop Conditions | `MaxMetricCallsStopper`, `TimeoutStopCondition`, `NoImprovementStopper`, `FileStopper`, `SignalStopper` |

### Pre-built Adapters (`adapters/`)

- **DefaultAdapter**: Single-turn LLM system prompt optimization
- **DSPyAdapter/DSPyFullProgramAdapter**: DSPy integration
- **GenericRAGAdapter**: Vector store-agnostic RAG optimization
- **MCPAdapter**: Model Context Protocol tool optimization
- **TerminalBenchAdapter**: Terminal-use agent optimization

### Generic Types

The codebase uses TypeVars (`DataInst`, `Trajectory`, `RolloutOutput`) allowing each adapter to define its own concrete types while the engine remains fully generic.

## Code Style

- Google Python Style Guide
- Ruff for linting/formatting (line length: 120)
- Relative imports banned (except in tests)
- Type checking with Pyright (standard mode)

## Development Guidelines

When modifying existing code or adding new features:

- **Breaking changes allowed**: Feel free to make breaking changes without considering backward compatibility.
- **Minimize changes**: Make the smallest possible modification that achieves the goal. Avoid refactoring unrelated code or adding unnecessary abstractions.
- **Follow existing patterns**: Study how similar functionality is implemented elsewhere in the codebase and mirror those conventions. For example:
