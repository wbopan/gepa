# Shared Active Subset & Coevolutionary Sample Selection

## Problem Statement

When optimizing with large datasets (M samples) but limited evaluation budget (N calls):

**Naive approach**: Each candidate evaluated on random subset of size b
- Different candidates tested on different subsets
- **Sparse coverage problem**: Hard to compare candidates directly
- Wasted budget on uninformative samples (too easy / too hard)

**Target**: Focus budget on the ~30% of samples that actually discriminate between candidates.

## Core Idea: Shared Active Subset

Instead of random sampling per candidate, maintain a **shared "exam" subset S** of size B:

```
1. Initialize: Sample B random examples from M to form S
2. Evaluate: All candidates in current generation evaluated on same S
3. Update S: After each generation
   - Remove samples where ALL candidates succeed (too easy)
   - Remove samples where ALL candidates fail (too hard)
   - Add new samples from M to maintain |S| = B
4. Repeat
```

**Why it works**:
- Dense comparison: All candidates share same test set
- Auto-focus: Discriminative samples (some pass, some fail) naturally retained
- Budget-optimal: Evaluation effort concentrated on informative samples

## Theoretical Foundation

### Optimal Subset Size

From the exploration-exploitation tradeoff analysis:

```
B ≈ N^(2/3)

Example: If N = 10,000 total budget
         Then B ≈ 464 samples in active subset
```

This balances:
- Larger B → more accurate comparison, fewer generations
- Smaller B → more exploration, more candidate evaluations

### Informativeness Criterion (Bucci & Pollack)

A test (sample) is **maximally informative** if it distinguishes between candidates:

```python
def is_informative(sample_id: DataId, candidate_scores: dict[ProgramIdx, float]) -> bool:
    """Sample is informative if different candidates get different scores."""
    scores = list(candidate_scores.values())
    return max(scores) > min(scores)  # Not all same
```

Samples where all candidates agree (all 0 or all 1) provide no gradient for selection.

## Related Techniques

### 1. Hall of Fame (Rosin & Belew, 1997)

Maintain archive of historically important tests/opponents:

```python
hall_of_fame: set[DataId] = set()

def update_hall_of_fame(sample_id: DataId, was_discriminative: bool):
    if was_discriminative:
        hall_of_fame.add(sample_id)
    # Periodically prune if too large
```

**Key insight**: Past discriminative samples may become informative again as candidates evolve.

### 2. Pareto Coevolution (Noble & Watson, 2001; Ficici & Pollack, 2001)

Treat each test as a separate objective dimension:

```python
# Instead of: score = mean(scores on all samples)
# Use: multi-objective = [score_on_sample_1, score_on_sample_2, ...]

# Pareto dominance: A dominates B iff A >= B on all samples, A > B on at least one
```

**GEPA already has this**: `frontier_type="instance"` tracks per-sample Pareto fronts.

### 3. Racing Algorithm (Maron & Moore, 1994)

Evaluate candidates in parallel, discard inferior ones early:

```python
def race(candidates: list, samples: Iterator) -> Candidate:
    """Return best candidate with statistical confidence."""
    scores = {c: [] for c in candidates}
    remaining = set(candidates)

    for sample in samples:
        for c in remaining:
            scores[c].append(evaluate(c, sample))

        # Hoeffding bound: discard candidates statistically worse than best
        best_mean = max(mean(scores[c]) for c in remaining)
        for c in list(remaining):
            n = len(scores[c])
            bound = sqrt(log(2/alpha) / (2*n))
            if mean(scores[c]) + bound < best_mean:
                remaining.remove(c)

        if len(remaining) == 1:
            break

    return remaining.pop()
```

### 4. DELPHI Algorithm (de Jong, 2004)

Identifies **Complete Evaluation Set** that provides ideal evaluation:

> "For any set of learners, a Complete Evaluation Set can be determined that provides ideal evaluation as specified by Evolutionary Multi-Objective Optimization."

## Applicability to GEPA

### Current State

GEPA already implements related concepts:

| GEPA Component | Coevolution Concept |
|----------------|---------------------|
| `pareto_front_valset` | Per-sample Pareto tracking |
| `PMaxBatchSampler` | Focus on hard/unattempted samples |
| `evaluation_cache` | Avoid redundant evaluations |

### Gap: Shared Subset Across Candidates

Current GEPA evaluates candidates on **different** minibatches each iteration.
Shared Active Subset requires evaluating **same** samples across candidates in a generation.

### Implementation Options

**Option A: InformativenessBatchSampler** (minimal change)

```python
@dataclass
class InformativenessBatchSampler(BatchSampler[DataId, DataInst]):
    """Sample based on cross-candidate score variance."""

    minibatch_size: int

    def next_minibatch_ids(self, loader, state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())

        # Compute informativeness: variance of scores across all programs
        informativeness = {}
        for data_id in all_ids:
            scores = [
                prog_scores.get(data_id, 0.0)
                for prog_scores in state.prog_candidate_val_subscores
                if data_id in prog_scores
            ]
            if len(scores) >= 2:
                mean = sum(scores) / len(scores)
                informativeness[data_id] = sum((s - mean)**2 for s in scores) / len(scores)
            else:
                informativeness[data_id] = 1.0  # Unexplored = high priority

        # Sample proportional to informativeness
        weights = [informativeness[did] + 0.1 for did in all_ids]  # +0.1 for exploration
        return random.choices(all_ids, weights=weights, k=self.minibatch_size)
```

**Option B: SharedActiveSubsetSampler** (moderate change)

```python
@dataclass
class SharedActiveSubsetSampler(BatchSampler[DataId, DataInst]):
    """Maintain fixed active subset, updated each generation."""

    subset_size: int = 200
    refresh_rate: float = 0.3  # Replace 30% non-discriminative samples per gen

    _active_subset: list[DataId] = field(default_factory=list)
    _generation_scores: dict[DataId, list[float]] = field(default_factory=dict)
    _last_generation: int = field(default=-1)

    def next_minibatch_ids(self, loader, state: GEPAState) -> list[DataId]:
        current_gen = state.i // self._candidates_per_gen  # Define generation boundary

        if current_gen > self._last_generation:
            self._update_subset(loader, state)
            self._last_generation = current_gen
            self._generation_scores.clear()

        return list(self._active_subset)

    def _update_subset(self, loader, state: GEPAState) -> None:
        """Replace non-discriminative samples."""
        all_ids = set(loader.all_ids())

        # Score each sample by variance in current generation
        to_remove = []
        for data_id in self._active_subset:
            scores = self._generation_scores.get(data_id, [])
            if len(scores) >= 2:
                variance = np.var(scores)
                if variance < 0.01:  # All candidates agree
                    to_remove.append(data_id)

        # Remove up to refresh_rate * subset_size samples
        n_remove = min(len(to_remove), int(self.refresh_rate * self.subset_size))
        for data_id in to_remove[:n_remove]:
            self._active_subset.remove(data_id)

        # Add new samples from pool
        pool = list(all_ids - set(self._active_subset))
        n_add = self.subset_size - len(self._active_subset)
        if pool and n_add > 0:
            self._active_subset.extend(random.sample(pool, min(n_add, len(pool))))
```

**Option C: Racing Early Stop** (engine change)

```python
# In core/engine.py, modify candidate evaluation:

def evaluate_with_racing(candidate, samples, best_score, alpha=0.05):
    """Evaluate until statistically confident or budget exhausted."""
    scores = []
    for sample in samples:
        scores.append(evaluate_one(candidate, sample))

        n = len(scores)
        mean = sum(scores) / n
        bound = math.sqrt(math.log(2/alpha) / (2*n))

        # Early stop if candidate is clearly worse
        if mean + bound < best_score:
            return None  # Reject early

        # Early stop if candidate is clearly better
        if mean - bound > best_score:
            break  # Accept early

    return mean
```

## Key References

### Foundational Papers

1. **Rosin & Belew (1997)** - Hall of Fame, Shared Sampling, Competitive Fitness Sharing
   - First systematic treatment of coevolutionary memory mechanisms

2. **Noble & Watson (2001)** - [Pareto Coevolution](https://www.researchgate.net/publication/2370567_Pareto_coevolution_Using_performance_against_coevolved_opponents_in_a_game_as_dimensions_for_Pareto_selection)
   - "Using performance against coevolved opponents in a game as dimensions for Pareto selection"
   - GECCO 2001

3. **Ficici & Pollack (2001)** - Pareto Optimality in Coevolutionary Learning
   - Developed novel coevolutionary algorithm based on Pareto optimality

4. **Bucci & Pollack (2002)** - [Order-theoretic Analysis of Coevolution](http://www.demo.cs.brandeis.edu/papers/bucci_foga_02.pdf)
   - Mathematical framework formalizing "maximally informative tests"
   - FOGA 2002

5. **Ficici (2004)** - Solution Concepts in Coevolutionary Algorithms
   - PhD thesis, Brandeis University
   - Comprehensive treatment of what "solution" means in coevolution

6. **de Jong (2004)** - DELPHI Algorithm
   - Guarantees monotonicity via Complete Evaluation Set

### Racing Algorithms

7. **Maron & Moore (1994)** - Hoeffding Races
   - "Accelerating model selection search for classification and function approximation"
   - NIPS 1994

8. **Maron & Moore (1997)** - [The Racing Algorithm](https://link.springer.com/article/10.1023/A:1006556606079)
   - "Model Selection for Lazy Learners"
   - Artificial Intelligence Review

9. **Birattari et al. (2002)** - [Racing for Metaheuristic Configuration](https://www.researchgate.net/publication/220740639_A_Racing_Algorithm_for_Configuring_Metaheuristics)
   - Applied racing to algorithm configuration
   - GECCO 2002

### Modern Curriculum Learning

10. **Zhang et al. (2020)** - Value Disagreement Sampling
    - "Automatic Curriculum Learning through Value Disagreement"
    - NeurIPS 2020

11. **Zhou & Bilmes (2020)** - [Dynamic Instance Hardness](https://proceedings.neurips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf)
    - "Curriculum Learning by Dynamic Instance Hardness"
    - NeurIPS 2020

### Surveys

12. **Popovici, Bucci, Wiegand & de Jong** - [Coevolutionary Principles](https://www.cs.tufts.edu/comp/150GA/handouts/nchb-main.pdf)
    - Comprehensive tutorial/book chapter on coevolution

## Summary

The "Shared Active Subset" approach combines:
- **Pareto Coevolution**: Treat samples as test cases that define objectives
- **Hall of Fame**: Maintain archive of historically discriminative samples
- **Racing**: Early-stop evaluation when statistical confidence achieved
- **Informativeness**: Prioritize samples with high cross-candidate variance

GEPA's existing infrastructure (Pareto fronts, evaluation cache, pluggable samplers) provides a solid foundation for implementing these techniques incrementally.
