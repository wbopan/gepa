# Papers on Diverse Expert Ensembles

## Overview

This document catalogs papers that focus on producing **a set of algorithms/policies each specialized in different domains**, rather than a single best algorithm. The key insight: maintaining diversity leads to better overall performance and robustness.

---

## Tier 1: Directly Relevant (LLM + Diversity)

### QDAIF: Quality-Diversity through AI Feedback
- **Venue**: ICLR 2024
- **Authors**: Herbie Bradley, Andrew Dai, et al.
- **URL**: https://qdaif.github.io/

**Core Contribution**:
- Uses LLM to evaluate BOTH quality AND diversity of solutions
- No hand-designed diversity measure needed
- LLM defines interpretable behavior dimensions (sentiment, style, etc.)

**Diversity Mechanism**:
```
Behavior Space = LLM-defined dimensions (e.g., tone, structure, approach)
Archive = MAP-Elites grid where each cell keeps best solution for that behavior
Result = Collection of high-quality solutions with diverse behaviors
```

**Why It Matters for GEPA**:
- Directly applicable: use LLM to evaluate prompt diversity
- No need to manually define what "different prompts" means

---

### ParetoPrompt: Pareto Prompt Optimization
- **Venue**: OpenReview 2024
- **URL**: https://openreview.net/forum?id=HGCk5aaSvE

**Core Contribution**:
- Multi-objective prompt optimization via RL
- No predefined objective weights
- **Goal: Set of prompts representing optimal trade-offs**

**Output**: A Pareto front of prompts, each optimal for different objective weightings (accuracy vs safety vs efficiency, etc.)

---

### FunSearch: Mathematical Discoveries via LLM
- **Venue**: Nature 2024
- **Authors**: DeepMind
- **URL**: https://www.nature.com/articles/s41586-023-06924-6

**Core Contribution**:
- LLM + evolutionary search discovers new mathematical constructions
- **Island model maintains diversity** across independent subpopulations

**Diversity Mechanism**:
```python
Island Model:
- 5+ independent populations evolving in parallel
- Migration: top solutions occasionally move between islands
- Reset: weak islands replaced with seeds from global best
- Result: multiple diverse lineages exploring different regions
```

**Key Insight**: Island isolation prevents premature convergence to single strategy.

---

### ELM: Evolution through Large Models
- **Venue**: Handbook of Evolutionary Machine Learning (2024)
- **Authors**: Lehman, Gordon, Jain, Ndousse, Yeh, Stanley
- **URL**: https://arxiv.org/abs/2206.08896

**Core Contribution**:
- LLM as intelligent mutation operator
- Combined with MAP-Elites for diverse code generation
- Generated **hundreds of thousands** of functional Python programs

**Output**: Archive of diverse robot controllers, each with different locomotion behaviors.

---

### EvoAgent: Automatic Multi-Agent Generation
- **Venue**: NAACL 2025
- **URL**: https://arxiv.org/abs/2406.14228

**Core Contribution**:
- Evolves single agent into **multi-agent system**
- Each agent specializes in different aspects
- Mutation/crossover/selection on agent configurations

**Output**: Diverse team of agents with complementary skills.

---

## Tier 2: Classical Diversity Methods

### MAP-Elites: Illuminating Search Spaces
- **Venue**: arXiv 2015
- **Authors**: Mouret & Clune
- **URL**: https://arxiv.org/abs/1504.04909

**The Foundation of QD**:
```
Input: Behavior space definition (e.g., robot gait parameters)
Process:
  1. Discretize behavior space into grid
  2. Each cell holds ONE elite (best in that behavior region)
  3. New solutions only compete within their cell
Output: Archive covering diverse behaviors, each locally optimal
```

**Application Example**: Robot learns 13,000+ different walking gaits; when damaged, quickly finds working alternative from archive.

---

### AlphaStar League
- **Venue**: Nature 2019
- **Authors**: DeepMind
- **URL**: https://www.nature.com/articles/s41586-019-1724-z

**Diversity Mechanism**: League Training
```
Agent Types:
- Main Agents: Primary learners
- Main Exploiters: Find weaknesses in main agents
- League Exploiters: Find weaknesses across entire league

Result: Population of diverse strategies that don't collapse to single meta
```

**Key Insight**: Exploiters explicitly maintain strategy diversity by rewarding finding weaknesses.

---

### Novelty Search
- **Venue**: Evolutionary Computation 2011
- **Authors**: Lehman & Stanley

**Radical Idea**: Completely abandon objective function
```
Fitness = Novelty (behavioral distance from archive)
NOT = Task performance

Result: Often finds task solutions FASTER than objective-driven search
        Plus discovers unexpected diverse solutions
```

**Why It Works**: Avoids deceptive gradients that lead to local optima.

---

### NSGA-III: Many-Objective Optimization
- **Venue**: IEEE TEVC 2014
- **Authors**: Deb & Jain

**For High-Dimensional Pareto Fronts**:
```
Problem: With >3 objectives, most solutions become non-dominated
Solution: Reference point method

Process:
  1. Generate uniformly distributed reference points on unit hyperplane
  2. Associate each solution with nearest reference point
  3. Selection prefers: (a) less crowded reference points, (b) closer to reference

Output: Pareto front with uniform spread across all objective trade-offs
```

---

## Tier 3: Supporting Techniques

### Crowding Distance (NSGA-II)
```python
def crowding_distance(front, objectives):
    """Prefer solutions in sparse regions of Pareto front."""
    distances = [0] * len(front)
    for obj in objectives:
        sorted_idx = argsort(front, key=obj)
        distances[sorted_idx[0]] = inf  # Boundaries always kept
        distances[sorted_idx[-1]] = inf
        for i in range(1, len(front)-1):
            distances[sorted_idx[i]] += (
                front[sorted_idx[i+1]][obj] - front[sorted_idx[i-1]][obj]
            ) / obj_range
    return distances
```

### MOEA/D Decomposition
```
Idea: Convert multi-objective problem into many single-objective subproblems
Each subproblem = weighted combination of objectives

Benefit: Weight vector distribution directly controls solution diversity
```

### Archive Management (SPEA2)
```
External archive stores non-dominated solutions
Truncation based on k-nearest-neighbor density
Removes solutions in crowded regions, preserves diversity
```

---

## Implementation Patterns for GEPA

### Pattern 1: Behavior-Based Archiving

```python
class BehaviorArchive:
    """MAP-Elites style archive for prompts."""

    def __init__(self, behavior_dims: list[str], resolution: int = 10):
        self.dims = behavior_dims
        self.resolution = resolution
        self.archive = {}  # (bin_tuple) -> (prompt, score)

    def compute_behavior(self, prompt: str) -> tuple[float, ...]:
        """Extract behavior features from prompt."""
        return (
            len(prompt) / 2000,  # length
            prompt.count('\n') / 20,  # structure
            len(re.findall(r'step \d|first|then|finally', prompt.lower())) / 10,  # reasoning
        )

    def discretize(self, behavior: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(min(int(b * self.resolution), self.resolution - 1) for b in behavior)

    def try_add(self, prompt: str, score: float) -> bool:
        """Add prompt if it's best in its behavior cell."""
        behavior = self.compute_behavior(prompt)
        bin_key = self.discretize(behavior)

        if bin_key not in self.archive or self.archive[bin_key][1] < score:
            self.archive[bin_key] = (prompt, score)
            return True
        return False

    def get_diverse_parents(self, n: int) -> list[str]:
        """Sample from different behavior regions."""
        cells = list(self.archive.keys())
        selected = random.sample(cells, min(n, len(cells)))
        return [self.archive[c][0] for c in selected]
```

### Pattern 2: Novelty-Quality Balance

```python
class NoveltyQualitySelector:
    """Balance exploration (novelty) and exploitation (quality)."""

    def __init__(self, novelty_weight: float = 0.3, k_neighbors: int = 5):
        self.novelty_weight = novelty_weight
        self.k = k_neighbors
        self.archive_embeddings = []

    def compute_novelty(self, embedding) -> float:
        if not self.archive_embeddings:
            return 1.0
        distances = [cosine_distance(embedding, a) for a in self.archive_embeddings]
        return np.mean(sorted(distances)[:self.k])

    def score(self, candidate_embedding, quality: float) -> float:
        novelty = self.compute_novelty(candidate_embedding)
        return (1 - self.novelty_weight) * quality + self.novelty_weight * novelty
```

### Pattern 3: Island Model

```python
class IslandEvolution:
    """Multiple independent populations with occasional migration."""

    def __init__(self, num_islands: int = 5):
        self.islands = [Population() for _ in range(num_islands)]
        self.generation = 0

    def evolve_step(self):
        # Independent evolution
        for island in self.islands:
            island.mutate_and_select()

        self.generation += 1

        # Periodic migration
        if self.generation % 10 == 0:
            self._migrate()

        # Reset stagnant islands
        if self.generation % 50 == 0:
            self._reset_weak()

    def _migrate(self):
        """Top candidate moves to random other island."""
        for i, island in enumerate(self.islands):
            if random.random() < 0.2:  # 20% migration rate
                target = random.choice([j for j in range(len(self.islands)) if j != i])
                self.islands[target].add(island.get_best())

    def _reset_weak(self):
        """Replace worst island with copy of best."""
        scores = [island.best_score for island in self.islands]
        worst, best = np.argmin(scores), np.argmax(scores)
        self.islands[worst] = self.islands[best].copy()

    def get_all_elites(self) -> list:
        """Return best from each island = diverse expert set."""
        return [island.get_best() for island in self.islands]
```

### Pattern 4: LLM-Based Diversity Evaluation (QDAIF Style)

```python
DIVERSITY_PROMPT = """
Analyze these two prompts and identify how they differ:

Prompt A:
{prompt_a}

Prompt B:
{prompt_b}

Rate their diversity on these dimensions (each 1-10):
1. Reasoning approach (step-by-step vs direct vs example-based)
2. Tone (formal vs casual vs instructional)
3. Structure (single paragraph vs numbered list vs dialogue)
4. Error handling (explicit vs implicit vs none)

Output JSON: {{"reasoning": N, "tone": N, "structure": N, "error_handling": N, "overall": N}}
"""

async def llm_diversity_score(prompt_a: str, prompt_b: str, llm) -> dict:
    response = await llm.generate(DIVERSITY_PROMPT.format(
        prompt_a=prompt_a, prompt_b=prompt_b
    ))
    return json.loads(response)
```

---

## Summary Table

| Paper | Year | Output Type | Diversity Mechanism | Relevance to GEPA |
|-------|------|-------------|---------------------|-------------------|
| QDAIF | 2024 | Diverse creative texts | LLM-defined behavior space | **High** - directly applicable |
| ParetoPrompt | 2024 | Pareto front of prompts | Multi-objective RL | **High** - same goal |
| FunSearch | 2024 | Mathematical algorithms | Island model | Medium - architecture pattern |
| ELM | 2022 | Diverse code | MAP-Elites + LLM | **High** - same approach |
| EvoAgent | 2025 | Multi-agent systems | Agent evolution | Medium - agent diversity |
| MAP-Elites | 2015 | Behavior archive | Grid discretization | **High** - foundational |
| AlphaStar | 2019 | Strategy league | Exploiter agents | Medium - game AI specific |
| NSGA-III | 2014 | Pareto front | Reference points | Medium - MOEA baseline |

---

## Key Takeaways

1. **Diversity is not just nice-to-have**: It often leads to better final solutions by avoiding local optima.

2. **Behavior space definition matters**: Whether hand-designed (MAP-Elites) or learned (AURORA) or LLM-evaluated (QDAIF).

3. **Island models are underutilized**: Simple to implement, effective at maintaining lineage diversity.

4. **LLMs enable new diversity measures**: Can evaluate semantic/behavioral diversity without hand-crafted features.

5. **GEPA already has foundations**: Pareto front tracking + LLM mutation = ready for QD enhancements.
