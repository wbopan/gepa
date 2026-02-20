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
- **Explore before designing**: Before designing any new feature or integration, thoroughly explore the repo to understand existing conventions — how experiments are structured (see `exps/`), whether to create a new adapter or reuse an existing one (e.g., `MemoryAdapter`), where prompts are defined, how datasets are loaded (`init_dataset()` patterns in `gepa.examples`), etc. Match the established patterns rather than inventing new abstractions.
- **Logging**: Use `get_logger()` from `gepa.logging` instead of Python's built-in `logging` or `print()`. Use `logger.log(msg, header="...")` for important messages with semantic headers (e.g., `"select"`, `"score"`, `"error"`), and `logger.debug()` for verbose output (enabled via `LOG_LEVEL=DEBUG`).
- **Experiment tracking**: Runs are logged to Weights & Biases (wandb). Use the MCP wandb tools to query run metrics and check experiment status (entity: `bmpixel`, project: `gepa-boost`). Before analyzing wandb data, read `knowledge/architecture/wandb_tables_and_metrics.md` for the full list of logged tables, metrics, and their semantics.

## Key Concepts

### Trainset vs Valset

**Minibatch IDs → trainset, not valset.** This is a common source of confusion.

- **Trainset**: Minibatch sampling for mutation feedback (`minibatch_outputs`, `train_sample_weights`)
- **Valset**: Full evaluation for Pareto tracking (`valset_outputs`, `pareto/val_candidates.*`)

See `knowledge/trainset_vs_valset.md` and `knowledge/wandb_tables_and_metrics.md` for details.

## Evolution Loop — High-Level Architecture

### Entry Point

`gepa.api.optimize()` → constructs `GEPAEngine` with all strategies → calls `engine.run()`.

### Initialization Phase (`engine.py:258-418`)

1. Evaluate seed candidate on full **valset** (`initialize_gepa_state` in `state.py:608-661`)
2. Initialize `GEPAState` with seed scores, Pareto frontier, lineage tracking
3. `state.i` starts at `-1`; seed eval budget counted via `state.total_num_evals = num_evals_run`
4. Publish seed prompt to weave (iteration=0), log seed metrics, fire `on_optimization_start` callback

### Main Loop (`engine.py:441-708`)

Each iteration:

```
state.i += 1  (0-indexed internally; displayed as state.i+1 in logs/callbacks)

1. [Reflective mutation]  — the primary proposal path
   │
   ├─ 1a. Select parent candidate
   │      CandidateSelector.select_candidate_idx(state)
   │      Implementation: strategies/candidate_selector.py
   │
   ├─ 1b. Sample minibatch from TRAINSET
   │      BatchSampler.next_minibatch_ids(trainset, state)
   │      Implementation: strategies/batch_sampler.py or strategies/adaboost_sampler.py
   │
   ├─ 1c. Evaluate parent on minibatch (capture_traces=True)
   │      adapter.evaluate(minibatch, curr_prog, capture_traces=True)
   │      Budget: state.increment_evals(len(subsample_ids))
   │      Implementation: proposer/reflective_mutation/common.py:145
   │
   ├─ 1d. Skip checks
   │      - No trajectories captured → skip
   │      - All scores perfect and skip_perfect_score=True → skip
   │
   ├─ 1e. Select components to update
   │      ReflectionComponentSelector(state, trajectories, scores, prog_id, prog)
   │      Implementation: strategies/component_selector.py (round_robin or all)
   │
   ├─ 1f. Build reflective dataset
   │      adapter.make_reflective_dataset(prog, eval_result, components_to_update)
   │      Returns: dict[component_name → list[{Inputs, Outputs, Feedback}]]
   │
   ├─ 1g. LLM proposes new instructions
   │      reflection_lm(prompt) for each component
   │      Implementation: strategies/instruction_proposal.py (InstructionProposalSignature)
   │      Or: adapter.propose_new_texts() if adapter provides it
   │
   ├─ 1h. Evaluate child on SAME minibatch (capture_traces=False)
   │      state.cached_evaluate_full(new_candidate, subsample_ids, ...)
   │      Budget: state.increment_evals(actual_evals_count)  (cache-aware)
   │      Implementation: reflective_mutation.py:187
   │
   └─ 1i. Acceptance gate
          if new_sum > old_sum → ACCEPT (strict improvement required)
          │  Full VALSET evaluation → state.update_state_with_new_program()
          │  Update Pareto frontier
          └  if new_sum <= old_sum → REJECT
```

### Acceptance and Pareto Update

When a child is accepted on minibatch (`engine.py:620`):

1. `_evaluate_on_valset()` — evaluate on valset (cache-aware), increment budget
2. `state.update_state_with_new_program()` — append candidate, update per-instance/objective/cartesian Pareto frontiers
3. Pareto frontier update logic (`state.py:434-530`):
   - If new score > current best for a val_id → replace frontier set
   - If new score == current best → add to frontier set (tie)
   - Dominated candidates pruned at selection time, not at update time

### Key Invariants

- **Minibatch from trainset, Pareto from valset** — never mixed
- **capture_traces=True only for parent eval** — child eval uses capture_traces=False
- **Strict improvement for mutation** (new_sum > old_sum)
- **Budget counts all evaluations**: parent minibatch + child minibatch + valset (if accepted)
- **Parent eval is never cached** (needs traces); child eval and valset eval use cache when enabled

### Feedback Descent Variant (`proposer/reflective_mutation/feedback_descent.py`)

Alternative to single-shot reflective mutation:
- Makes up to `max_attempts` (default 3) refinement passes on same minibatch
- Each failed attempt's feedback is accumulated into the next prompt
- Returns the best attempt (even if no improvement over parent)
- Engine still applies the same acceptance gate (new_sum > old_sum)

## Configurable Options Reference

### Candidate Selection (`candidate_selection_strategy`)

| Strategy | Implementation | Description |
|----------|---------------|-------------|
| `"pareto"` (default) | `ParetoCandidateSelector` | Sample from Pareto frontier proportional to #wins |
| `"current_best"` | `CurrentBestCandidateSelector` | Always select highest-scoring candidate |
| `"epsilon_greedy"` | `EpsilonGreedyCandidateSelector` | ε-greedy exploration (ε=0.1) |
| `"avg_family"` | `AvgFamilyScoreCandidateSelector` | Select by average family (lineage) score |
| `"max_family"` | `MaxFamilyScoreCandidateSelector` | Select by max family score |

### Batch Sampling (`batch_sampler`)

| Strategy | Implementation | Description |
|----------|---------------|-------------|
| `"epoch_shuffled"` (default) | `EpochShuffledBatchSampler` | Shuffle all IDs each epoch, sample sequentially |
| `"adaboost"` | `AdaBoostBatchSampler` | Weight samples by past performance (harder examples sampled more) |
| (instance) | `BayesianBatchSampler` | Variance-based sampling; prioritizes uncertain samples with balanced success/failure ratios |
| (instance) | `PMaxBatchSampler` | AdaBoost variant that tracks best score per sample and boosts unattempted ones |

### Frontier Type (`frontier_type`)

| Type | Description |
|------|-------------|
| `"instance"` (default) | Best score per validation example |
| `"objective"` | Best score per objective metric (requires multi-objective evaluator) |
| `"hybrid"` | Both instance and objective frontiers |
| `"cartesian"` | Per (example, objective) pair |

### Proposer Type (`proposer_type`)

| Type | Description |
|------|-------------|
| `"reflective_mutation"` (default) | Single-shot LLM reflection and mutation |
| `"feedback_descent"` | Iterative refinement with accumulated failure feedback |

### Component Selection (`module_selector`)

| Strategy | Description |
|----------|-------------|
| `"round_robin"` (default) | Cycle through components in order |
| `"all"` | Update all components every iteration |

### Stop Conditions

- `max_metric_calls`: Budget cap on total evaluations (minibatch + valset)
- `stop_callbacks`: Custom `StopperProtocol` instances (FileStopper, TimeoutStopCondition, SignalStopper, NoImprovementStopper)
- `run_dir` + `gepa.stop` file: Graceful stop via file presence

### Iteration Numbering Convention

- `state.i`: 0-indexed (starts at -1, incremented to 0 before first iteration)
- Callbacks: use `state.i + 1` (1-indexed, human-readable)
- W&B metrics: use `state.i` (0-indexed)
- Seed evaluation: logged at `state.i = -1` for metrics, `iteration=0` for weave prompts
