# Trainset vs Valset in GEPA

## Overview

GEPA uses two distinct datasets during optimization:

| Aspect | Trainset | Valset |
|--------|----------|--------|
| **Purpose** | Feedback loop for mutation | Performance measurement |
| **Required** | Yes | Optional (defaults to trainset) |
| **Evaluation** | Minibatch sampling | Full evaluation |
| **Goal** | Rapid iteration & exploration | Rigorous comparison & Pareto tracking |

## API Definition

```python
def optimize(
    trainset: list[DataInst] | DataLoader[DataId, DataInst],  # Required
    valset: list[DataInst] | DataLoader[DataId, DataInst] | None = None,  # Optional
    ...
)
```

If `valset` is not provided, GEPA reuses trainset for validation.

## Usage in Optimization Loop

### Trainset (Reflection & Mutation)

Located in `src/gepa/proposer/reflective_mutation/reflective_mutation.py`:

1. Each iteration samples a **minibatch** from trainset
2. Current candidate is evaluated on minibatch to capture **traces/feedback**
3. Feedback is used by the LLM to **reflect and propose mutations**
4. New candidate is quickly tested on the **same minibatch** for accept/reject decision

```python
# Minibatch sampling (line 136)
subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
minibatch = self.trainset.fetch(subsample_ids)

# Evaluation with trace capture (line 186)
eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
```

### Valset (Performance Tracking)

Located in `src/gepa/core/engine.py`, method `_evaluate_on_valset()`:

1. After a candidate passes the minibatch test, it gets a **full valset evaluation**
2. Valset scores determine:
   - **Pareto frontier** (best candidates per validation example)
   - **Aggregate score** (overall candidate quality)
   - **Final best program** selection

```python
# Full valset evaluation (line 155)
valset_evaluation = self._evaluate_on_valset(new_program, state)
```

## Metrics Tracked (wandb)

### Trainset Metrics

| Metric | Description |
|--------|-------------|
| `train/batch_score_before` | Score on minibatch before mutation |
| `train/batch_score_after` | Score on minibatch after mutation |
| `minibatch_outputs` | Table of predictions on current minibatch |

### Valset Metrics

| Metric | Description |
|--------|-------------|
| `val/new_score` | New candidate's validation score |
| `val/best_agg` | Best aggregate validation score achieved |
| `val/eval_count` | Number of validation samples evaluated |
| `pareto/val_agg` | Aggregated Pareto validation score |
| `pareto/val_candidates.*` | Per-sample scores for Pareto tracking |
| `valset_outputs` | Table of predictions on full validation set |

## State Persistence

Valset results are persisted in the optimization state:

- `prog_candidate_val_subscores`: Dict mapping validation example ID to score per candidate
- `program_at_pareto_front_valset`: Best candidates per validation example (Pareto frontier)

## Example Configuration

```python
# From src/gepa/examples/gpqa.py
trainset = all_examples[:98]    # 98 examples for mutation feedback
valset = all_examples[-100:]    # 100 examples for performance tracking
```

## Design Rationale

- **Trainset**: Low-cost, fast feedback for rapid iteration. Minibatch sampling reduces evaluation cost while still providing useful improvement signals.
- **Valset**: Comprehensive, unbiased measurement. Full evaluation ensures accurate performance tracking and prevents overfitting to training patterns.

This separation allows GEPA to balance exploration efficiency (cheap trainset minibatches) with evaluation rigor (comprehensive valset assessment).
