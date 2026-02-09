# Diversity Optimization in Evolutionary Algorithms

## Overview

This document surveys methods for producing **diverse sets of high-quality solutions** rather than a single optimum. Three main paradigms exist:

| Paradigm | Core Idea | Key Algorithms |
|----------|-----------|----------------|
| **Quality-Diversity (QD)** | Grid behavior space, keep elite per cell | MAP-Elites, QDAIF |
| **Multi-Objective EA (MOEA)** | Maintain Pareto front spread | NSGA-II/III, MOEA/D |
| **Island Models** | Multiple independent subpopulations | FunSearch |

---

## Part 1: Quality-Diversity (QD) Algorithms

### Core Concept

QD optimization seeks a **collection of diverse, high-performing solutions** rather than a single best. This is valuable when:
- Environment may change (robot damage recovery)
- Different scenarios need different strategies
- Creativity/exploration is inherently valuable

### MAP-Elites (2015)

**Paper**: Mouret & Clune, "Illuminating search spaces by mapping elites", arXiv:1504.04909

**Mechanism**:
```
1. Discretize behavior space into grid cells
2. Each cell holds ONE elite (best solution in that behavior region)
3. New solutions compete only with their cell's current elite
4. Result: Archive of diverse high-quality solutions
```

**Key Properties**:
- Simple, effective, parallelizable
- Often finds better global optima than traditional methods (via diversity)
- Requires predefined behavior descriptors

**Applications**:
- Meta's LLama v3: Finding adversarial prompts via MAP-Elites
- Robot damage recovery (Nature 2015): Learning hundreds of walking gaits

### Novelty Search (2008)

**Paper**: Lehman & Stanley, "Exploiting Open-Endedness to Solve Problems Through the Search for Novelty"

**Core Insight**: Most ambitious objectives don't illuminate the path toward themselves.

**Mechanism**:
- **Completely abandon objective function**
- Reward only behavioral novelty (distance from archive)
- Counterintuitively, often finds objectives faster than objective-driven search

### CMA-MAE (2022)

**Paper**: Fontaine & Nikolaidis, "Covariance Matrix Adaptation MAP-Annealing", arXiv:2205.10752

**Improvement over CMA-ME**:
- Better balance between exploration and exploitation
- Handles flat objective functions
- Works with low-resolution archives

**Implementation**: [pyribs](https://pyribs.org/)

### AURORA (2021)

**Paper**: Cully, "Unsupervised Behaviour Discovery with Quality-Diversity Optimisation", arXiv:2106.05648

**Key Innovation**: **Automatically learn behavior descriptors** using autoencoders
- No need for domain expert to define behavior space
- Learns latent space from raw sensor data
- Enables discovery of unexpected behavioral dimensions

### QDAIF (ICLR 2024)

**Paper**: Bradley et al., "Quality-Diversity through AI Feedback"

**Key Innovation**: Use LLM to evaluate both quality AND diversity
- No hand-designed diversity measure needed
- LLM defines behavior dimensions (sentiment, style, ending type, etc.)
- Validated on creative writing tasks

**Highly relevant to GEPA**: Shows LLMs can effectively work within QD frameworks.

---

## Part 2: Multi-Objective Evolutionary Algorithms (MOEA)

### NSGA-II (2002)

**Paper**: Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II"

**Core Mechanisms**:
1. **Fast non-dominated sorting**: O(MN²) complexity
2. **Crowding distance**: Measures solution density, prefers sparse regions
3. **Elitism**: Non-dominated solutions survive to next generation

**Crowding Distance Calculation**:
```python
def crowding_distance(solutions, objectives):
    n = len(solutions)
    distances = [0.0] * n

    for obj_idx in range(num_objectives):
        sorted_idx = sorted(range(n), key=lambda i: solutions[i][obj_idx])
        distances[sorted_idx[0]] = float('inf')   # Boundary solutions
        distances[sorted_idx[-1]] = float('inf')

        obj_range = solutions[sorted_idx[-1]][obj_idx] - solutions[sorted_idx[0]][obj_idx]
        if obj_range > 0:
            for i in range(1, n-1):
                distances[sorted_idx[i]] += (
                    solutions[sorted_idx[i+1]][obj_idx] -
                    solutions[sorted_idx[i-1]][obj_idx]
                ) / obj_range

    return distances
```

**Limitation**: Performance degrades with >3 objectives (Pareto resistance).

### NSGA-III (2014)

**Designed for**: Many-objective optimization (>3 objectives)

**Key Change**: Replace crowding distance with **reference points**
- Das-Dennis method generates uniformly distributed reference points
- Solutions associate with nearest reference point
- Ensures coverage across entire Pareto front

**Adaptive Variants**:
- A-NSGA-III: Dynamically adjust reference vectors for irregular fronts
- NSGA-III-UR: Selectively activate reference vector adaptation

### MOEA/D (2007)

**Paper**: Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"

**Core Idea**: Decompose multi-objective problem into scalar subproblems
- Each subproblem has a weight vector
- Neighboring subproblems share information
- Uniform weight vectors → uniform Pareto front coverage

**Diversity Mechanism**: Weight vector distribution naturally ensures diversity.

### SPEA2 (2001)

**Paper**: Zitzler et al., "SPEA2: Improving the Strength Pareto Evolutionary Algorithm"

**Key Features**:
- External archive stores non-dominated solutions
- Fine-grained fitness: strength value + k-nearest neighbor density
- Truncation preserves diversity without extra parameters

### Pareto Front Quality Metrics

| Metric | Measures | Properties |
|--------|----------|------------|
| **Hypervolume (HV)** | Volume dominated by solution set | Pareto-compliant, expensive |
| **IGD** | Distance from true front to approximation | Needs true front |
| **IGD+** | Weakly Pareto-compliant IGD | Better theoretical properties |
| **Spread (Δ)** | Distribution range | Pure diversity |
| **Spacing** | Distribution uniformity | Pure uniformity |

---

## Part 3: LLM + Evolutionary Algorithms

### EvoPrompt (ICLR 2024)

**Paper**: "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers"

**Mechanism**:
- GA and DE algorithms with LLM as mutation operator
- LLM generates coherent, human-readable mutations

**Diversity Handling**: Limited (mainly via DE's exploration capability)

### PromptBreeder (ICML 2024)

**Paper**: "Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution"

**Key Innovation**: **Self-referential evolution**
- Evolve both task-prompts AND mutation-prompts
- Meta-learning loop: improve "how to improve prompts"

### ParetoPrompt (2024)

**Paper**: "Pareto Prompt Optimization"

**Key Innovation**: RL-based multi-objective prompt optimization
- No predefined objective weights
- Explores entire Pareto front
- **Goal: Set of prompts representing best trade-offs**

### FunSearch (Nature 2024)

**Paper**: "Mathematical discoveries from program search with large language models"

**Diversity Mechanism**: **Island Model**
```python
class IslandEvolution:
    def __init__(self, num_islands=5):
        self.islands = [Population() for _ in range(num_islands)]

    def step(self):
        # Independent evolution
        for island in self.islands:
            island.evolve()

        # Migration: probabilistic based on fitness
        self.migrate_between_islands()

        # Reset: replace weak islands with global best
        self.reset_weak_islands()
```

**Result**: Discovered new mathematical constructions surpassing known optima.

### ELM - Evolution through Large Models (2022)

**Paper**: Lehman et al., "Evolution through Large Models", arXiv:2206.08896

**Key Innovation**: LLM encodes human knowledge for intelligent mutation
- Combined with MAP-Elites for diverse code generation
- Generated hundreds of thousands of functional Python programs

**Open Source**: [OpenELM](https://github.com/CarperAI/OpenELM)

### EvoAgent (NAACL 2025)

**Paper**: "EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms"

**Key Innovation**: Evolve single agent into multi-agent system
- Mutation, crossover, selection operators on agents
- Generates agents with diverse configurations

### OMNI-EPIC (2024)

**Paper**: "OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness"

**Key Innovation**: LLM-based "interestingness model"
- Generates infinite stream of learnable, interesting tasks
- Uses similar tasks as stepping stones
- Open-ended learning paradigm

---

## Part 4: Implications for GEPA

### Current GEPA Mechanisms

GEPA already has foundational QD/MOEA elements:

| GEPA Component | QD/MOEA Analog |
|----------------|----------------|
| `frontier_type: cartesian` | Multi-dimensional behavior space |
| `ParetoCandidateSelector` | Pareto-based parent selection |
| `ReflectiveMutationProposer` | Intelligent LLM mutation (like ELM) |
| `MergeProposer` | Crossover operator |

### Recommended Enhancements

#### 1. Explicit Behavior Descriptors

Define prompt phenotype features for MAP-Elites-style archiving:

```python
def compute_behavior_descriptor(candidate: dict[str, str],
                                 trajectories: list[Trajectory]) -> tuple[float, ...]:
    """Extract behavior features from prompt and its execution."""
    features = (
        len(candidate["prompt"]) / 1000,  # Normalized length
        count_reasoning_steps(candidate["prompt"]),  # Reasoning depth
        count_examples(candidate["prompt"]),  # Few-shot count
        # ... additional features
    )
    return features
```

#### 2. Novelty Reward in Candidate Selection

```python
class NoveltyAwareCandidateSelector:
    def __init__(self, archive: list, k_neighbors: int = 5, novelty_weight: float = 0.3):
        self.archive = archive
        self.k = k_neighbors
        self.novelty_weight = novelty_weight

    def compute_novelty(self, candidate_embedding) -> float:
        """Average distance to k nearest neighbors in archive."""
        distances = [cosine_distance(candidate_embedding, a) for a in self.archive]
        return np.mean(sorted(distances)[:self.k])

    def select(self, candidates, quality_scores):
        novelty_scores = [self.compute_novelty(embed(c)) for c in candidates]
        combined = [q + self.novelty_weight * n
                    for q, n in zip(quality_scores, novelty_scores)]
        return weighted_sample(candidates, combined)
```

#### 3. Island Model Evolution

```python
class IslandBasedGEPA:
    """Multiple independent populations with migration."""

    def __init__(self, num_islands: int = 5, migration_interval: int = 10):
        self.islands = [GEPAState() for _ in range(num_islands)]
        self.migration_interval = migration_interval

    def migrate(self):
        """Exchange top candidates between adjacent islands."""
        for i in range(len(self.islands)):
            source = self.islands[i]
            target = self.islands[(i + 1) % len(self.islands)]
            # Probabilistically migrate best candidate
            if random.random() < 0.1:  # 10% migration rate
                best = source.get_best_candidate()
                target.add_candidate(best)

    def reset_weak_island(self):
        """Replace worst island with global best."""
        worst_idx = min(range(len(self.islands)),
                        key=lambda i: self.islands[i].best_score)
        global_best = max(self.islands, key=lambda s: s.best_score).get_best_candidate()
        self.islands[worst_idx] = GEPAState(initial_candidate=global_best)
```

#### 4. QDAIF-Style Diversity Evaluation

```python
DIVERSITY_EVAL_PROMPT = """
Compare these two prompts and rate their diversity (1-10):

Prompt A: {prompt_a}
Prompt B: {prompt_b}

Consider:
- Reasoning approach (step-by-step vs direct)
- Instruction style (formal vs conversational)
- Example selection strategy
- Error handling approach

Respond with just the number.
"""

async def evaluate_diversity(prompt_a: str, prompt_b: str, llm) -> float:
    response = await llm.generate(
        DIVERSITY_EVAL_PROMPT.format(prompt_a=prompt_a, prompt_b=prompt_b)
    )
    return float(response.strip()) / 10.0
```

#### 5. Crowding Distance for Pareto Selection

```python
def crowding_distance_selection(pareto_front: list[Candidate],
                                  scores: dict[str, list[float]],
                                  n_select: int) -> list[Candidate]:
    """Select candidates from Pareto front preferring sparse regions."""
    distances = compute_crowding_distance(pareto_front, scores)

    # Combine with quality for selection probability
    probs = np.array(distances)
    probs = probs / probs.sum()

    selected_idx = np.random.choice(len(pareto_front), size=n_select,
                                     replace=False, p=probs)
    return [pareto_front[i] for i in selected_idx]
```

---

## Key References

### Quality-Diversity
- Mouret & Clune (2015): MAP-Elites - arXiv:1504.04909
- Lehman & Stanley (2011): Novelty Search - Evolutionary Computation
- Cully et al. (2015): Robots that adapt like animals - Nature
- Bradley et al. (2024): QDAIF - ICLR 2024
- Fontaine & Nikolaidis (2022): CMA-MAE - arXiv:2205.10752

### Multi-Objective EA
- Deb et al. (2002): NSGA-II - IEEE TEVC
- Zhang & Li (2007): MOEA/D - IEEE TEVC
- Zitzler et al. (2001): SPEA2 - ETH Technical Report

### LLM + Evolution
- Romera-Paredes et al. (2024): FunSearch - Nature
- Lehman et al. (2022): ELM - arXiv:2206.08896
- Bradley et al. (2024): QDAIF - ICLR 2024
- (2024): EvoPrompt - ICLR 2024
- (2024): PromptBreeder - ICML 2024
- (2024): ParetoPrompt - OpenReview

### Tools & Libraries
- [pyribs](https://pyribs.org/): QD algorithms (CMA-ME, CMA-MAE)
- [QDax](https://github.com/adaptive-intelligent-robotics/QDax): JAX-based QD
- [OpenELM](https://github.com/CarperAI/OpenELM): LLM + MAP-Elites
- [pymoo](https://pymoo.org/): Multi-objective optimization
