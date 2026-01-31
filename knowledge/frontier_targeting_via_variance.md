# Frontier Targeting via Variance

## Problem

Standard difficulty-based sampling uses score as the sole signal:
- Low score → high weight (sample more)
- High score → low weight (sample less)

This doesn't distinguish between:
- **Too hard**: Consistently unsolvable (score always 0)
- **Frontier**: Sometimes solvable, sometimes not (high variance)
- **Too easy**: Consistently solved (score always 1)

The **frontier** samples provide the strongest learning signal, but naive methods waste compute on truly unsolvable samples.

## Core Insight

From Value Disagreement Sampling (VDS, NeurIPS 2020):
> Goals at the "frontier" of the agent's capabilities provide the strongest learning signal.

Key observation:
- **High score variance** across evaluations = frontier sample
- **Low variance, low mean** = too hard (waste of compute)
- **Low variance, high mean** = too easy (already mastered)

## Frontier Score Formula

```python
def frontier_score(scores: list[float]) -> float:
    """Higher = more likely to be at the frontier."""
    if len(scores) < 2:
        return 0.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    # Peak priority at mean=0.5 (most uncertain), decay towards 0 and 1
    mean_factor = 4 * mean * (1 - mean)  # Parabola: 0 at edges, 1 at center

    # Variance boost: high variance = more interesting
    var_factor = min(variance * 4, 1.0)  # var of 0.25 is max for [0,1]

    return 0.5 * mean_factor + 0.5 * var_factor
```

### Alternative Formulas

```python
# Pure variance (simplest)
frontier_score = variance

# Entropy-based (information-theoretic)
frontier_score = -m * log(m) - (1-m) * log(1-m) if 0 < m < 1 else 0

# TD-error analog (change over time)
frontier_score = abs(scores[-1] - scores[-2]) if len(scores) >= 2 else 0
```

## Key Design Decisions

### 1. Window Size
- Too small: Noisy variance estimates
- Too large: Slow to adapt as model improves
- Recommended: 5-10 evaluations

### 2. Formula Balance
`0.5 * mean_factor + 0.5 * var_factor` balances:
- **Mean factor**: Samples with ~50% success rate are inherently uncertain
- **Variance factor**: High variance indicates the model is "on the edge"

### 3. Exploration Weight
Mix frontier priority with uniform exploration:
```python
weight = (1 - exploration_weight) * frontier_score + exploration_weight
```
- Set to 0: Pure frontier exploitation (may miss newly-solvable samples)
- Set to 1: Uniform random (baseline)
- Recommended: 0.2-0.4 for balance

## References

- **Value Disagreement Sampling (VDS)**: Zhang et al., "Automatic Curriculum Learning through Value Disagreement", NeurIPS 2020
- **Dynamic Instance Hardness (DIHCL)**: Zhou & Bilmes, "Curriculum Learning by Dynamic Instance Hardness", NeurIPS 2020
- **Prioritized Experience Replay**: Schaul et al., 2015 - uses TD-error as priority signal
