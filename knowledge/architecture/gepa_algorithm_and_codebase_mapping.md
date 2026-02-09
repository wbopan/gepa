# GEPA Algorithm and Codebase Mapping

This document describes the GEPA (Genetic-Pareto) algorithm from the paper "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (arXiv:2507.19457) and maps each component to the corresponding code in this repository.

## Algorithm Overview

GEPA is a **reflective prompt optimizer** for compound AI systems that combines:
1. **Genetic prompt evolution** - iteratively mutating and selecting prompts
2. **Natural language reflection** - using LLM feedback to diagnose problems and propose improvements
3. **Pareto-based candidate selection** - maintaining diverse strategies via multi-objective optimization

### Key Insight

Unlike RL methods (e.g., GRPO) that derive policy gradients from sparse scalar rewards, GEPA leverages the **interpretable nature of language**. System trajectories (reasoning chains, tool calls, error messages) can be serialized into text that LLMs can reflect upon, enabling much more sample-efficient learning.

### Results Summary

- Outperforms GRPO by 10% average (up to 20%) while using up to **35x fewer rollouts**
- Outperforms MIPROv2 by over 10% across benchmarks
- Generates shorter, more efficient prompts (up to 9.2x shorter than MIPROv2)

---

## Algorithm Flow

### High-Level Pseudocode (from Paper Algorithm 1)

```
Input: System Φ, train set D_train, metric μ, feedback function μ_f, budget B

1. Initialize candidate pool P with base system parameters
2. Evaluate initial candidate on validation set D_pareto
3. While budget not exhausted:
   a. Select candidate from pool (Pareto-based selection)
   b. Propose new candidate via:
      - Reflective Mutation (primary), OR
      - System-Aware Merge (secondary)
   c. Evaluate on minibatch
   d. If improved: add to pool, evaluate on full validation set
   e. Update Pareto frontier
4. Return best candidate by aggregate validation score
```

---

## Component-to-Code Mapping

### 1. Main Optimization Loop

**Paper Section:** 3.1 Genetic Optimization Loop

**Code:** `core/engine.py` → `GEPAEngine.run()`

```python
class GEPAEngine:
    def run(self) -> GEPAState:
        # Initialize state with seed candidate
        state = initialize_gepa_state(...)

        while not self._should_stop(state):
            # Save checkpoint
            state.save(self.run_dir)

            # Try merge (if enabled and conditions met)
            if self.merge_proposer and self._should_try_merge():
                proposal = self.merge_proposer.propose(state)
                if self._accept(proposal):
                    self._run_full_eval_and_add(state, proposal)

            # Reflective mutation (primary strategy)
            proposal = self.reflective_mutation_proposer.propose(state)

            # Accept if minibatch score improves
            if proposal.new_subsample_score > proposal.old_subsample_score:
                self._run_full_eval_and_add(state, proposal)

            # Fire callbacks, check stopping conditions
            self._notify_callbacks(...)

        return state
```

**Key Design:**
- Accepts candidates based on **minibatch improvement** (sample efficient)
- Only runs full validation evaluation **after acceptance** (saves rollouts)
- Supports checkpointing and resumption

---

### 2. Reflective Prompt Mutation

**Paper Section:** 3.2 Reflective Prompt Mutation

**Code:** `proposer/reflective_mutation/reflective_mutation.py` → `ReflectiveMutationProposer`

**Algorithm:**
1. Select a candidate to mutate
2. Select target module(s) via round-robin
3. Sample minibatch from training set
4. Execute system with `capture_traces=True`
5. Build reflective dataset from trajectories
6. Use LLM to propose new instructions
7. Evaluate mutated candidate on same minibatch

```python
class ReflectiveMutationProposer:
    def propose(self, state: GEPAState) -> CandidateProposal:
        # Step 1: Select candidate (Pareto-based)
        candidate_idx = self.candidate_selector.select_candidate_idx(state)
        candidate = state.program_candidates[candidate_idx]

        # Step 2: Sample minibatch
        minibatch_ids = self.batch_sampler.next_minibatch_ids()
        batch = [self.fetcher(id) for id in minibatch_ids]

        # Step 3: Evaluate with trace capture
        eval_batch = self.adapter.evaluate(
            batch, candidate, capture_traces=True
        )

        # Step 4: Select components to update
        components = self.component_selector.select_components(candidate)

        # Step 5: Build reflective dataset
        reflective_dataset = self.adapter.make_reflective_dataset(
            candidate, eval_batch, components
        )

        # Step 6: LLM proposes new texts
        new_texts = self.propose_new_texts(
            candidate, reflective_dataset, components
        )

        # Step 7: Evaluate mutant
        new_candidate = {**candidate, **new_texts}
        new_eval = self.adapter.evaluate(
            batch, new_candidate, capture_traces=False
        )

        return CandidateProposal(
            new_candidate=new_candidate,
            old_subsample_score=mean(eval_batch.scores),
            new_subsample_score=mean(new_eval.scores),
            parent_idx=candidate_idx
        )
```

**Reflective Dataset Format:**
```python
{
    "component_name": [
        {
            "Inputs": "The query passed to this module",
            "Outputs": "What the module produced",
            "Feedback": "Success/failure analysis, error messages, etc."
        },
        # ... more examples from minibatch
    ]
}
```

**Feedback Function (μ_f):**
The paper emphasizes using **evaluation traces** (compiler errors, profiling results, etc.) as diagnostic signals. This is implemented via the adapter's `make_reflective_dataset()` which can include rich textual feedback beyond just pass/fail.

---

### 3. Pareto-Based Candidate Selection

**Paper Section:** 3.3 Pareto-based Candidate Selection

**Code:** `strategies/candidate_selector.py` → `ParetoCandidateSelector`

**Why Pareto Selection?**
- Naive "select best candidate" leads to local optima
- System exhausts budget trying to improve one dominant strategy
- Pareto selection maintains diversity by tracking best-per-instance

**Algorithm (from Paper Algorithm 2):**
1. For each validation instance, find the highest score achieved by any candidate
2. Compile list of candidates that achieve best score on at least one instance
3. Prune strictly dominated candidates
4. Sample with probability proportional to number of "wins"

```python
class ParetoCandidateSelector:
    def select_candidate_idx(self, state: GEPAState) -> int:
        # Get Pareto frontier: val_id -> best_score
        pareto_front = state.pareto_front_valset

        # Get candidates on frontier: val_id -> set of candidate indices
        frontier_programs = state.program_at_pareto_front_valset

        # Count wins per candidate
        candidate_wins = defaultdict(int)
        for val_id, program_indices in frontier_programs.items():
            for idx in program_indices:
                candidate_wins[idx] += 1

        # Weighted random selection
        candidates = list(candidate_wins.keys())
        weights = [candidate_wins[c] for c in candidates]
        return random.choices(candidates, weights=weights)[0]
```

**State Tracking:** `core/state.py` → `GEPAState`

```python
class GEPAState:
    # Core Pareto data structures
    pareto_front_valset: dict[str, float]           # val_id -> best_score
    program_at_pareto_front_valset: dict[str, set]  # val_id -> {candidate_indices}

    # Candidate lineage (for merge)
    parent_program_for_candidate: dict[int, int]    # child_idx -> parent_idx

    def update_state_with_new_program(self, candidate, val_scores, ...):
        """Update Pareto frontier after new candidate is accepted."""
        for val_id, score in val_scores.items():
            current_best = self.pareto_front_valset.get(val_id, float('-inf'))

            if score > current_best:
                # New best - replace frontier
                self.pareto_front_valset[val_id] = score
                self.program_at_pareto_front_valset[val_id] = {new_idx}
            elif score == current_best:
                # Tie - add to frontier
                self.program_at_pareto_front_valset[val_id].add(new_idx)
```

**Frontier Types:**
The code supports multiple frontier strategies via `frontier_type` parameter:
- `instance` (default): Best score per validation instance
- `objective`: Best score per objective metric (multi-objective)
- `hybrid`: Both instance and objective frontiers
- `cartesian`: Per (instance, objective) pair

---

### 4. System-Aware Merge (Crossover)

**Paper Section:** Appendix F

**Code:** `proposer/merge.py` → `MergeProposer`

**Purpose:** Combine complementary strategies from different evolutionary branches.

**Algorithm:**
1. Select two candidates from Pareto frontier
2. Find their common ancestor in lineage tree
3. Create merged candidate by picking best components from each parent
4. Evaluate on intersection of their successful validation instances

```python
class MergeProposer:
    def propose(self, state: GEPAState) -> Optional[CandidateProposal]:
        # Select two frontier candidates
        parent1_idx, parent2_idx = self._select_two_from_frontier(state)

        # Find common ancestor
        ancestor_idx = self._find_common_ancestor(
            state.parent_program_for_candidate,
            parent1_idx, parent2_idx
        )

        # Merge: pick best version of each component
        merged = self._merge_components(
            state.program_candidates[parent1_idx],
            state.program_candidates[parent2_idx],
            state.program_candidates[ancestor_idx]
        )

        # Evaluate on common validation instances
        common_val_ids = self._get_common_wins(parent1_idx, parent2_idx, state)
        eval_result = self._evaluate_on_subset(merged, common_val_ids)

        return CandidateProposal(new_candidate=merged, ...)
```

**When to Merge:**
- Only after successful reflective mutation (found new strategy)
- Limited invocations per run (`max_merge_invocations`)
- Paper notes optimal timing needs further study

---

### 5. Adapter Protocol (System Integration)

**Paper Section:** 2 Problem Statement

**Code:** `core/adapter.py` → `GEPAAdapter` (Protocol)

The adapter is the **single integration point** for any AI system:

```python
class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Execute candidate on batch, return scores and optional trajectories.

        Args:
            batch: List of data instances
            candidate: Dict mapping component_name -> prompt_text
            capture_traces: Whether to record execution traces

        Returns:
            EvaluationBatch with:
            - outputs: List of system outputs
            - scores: List of scalar scores (higher = better)
            - trajectories: Execution traces (only if capture_traces=True)
            - objective_scores: Optional multi-objective scores
        """

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str]
    ) -> dict[str, list[dict]]:
        """
        Extract textual information from trajectories for reflection.

        Returns dict: component_name -> list of records with:
        - Inputs: What the module received
        - Outputs: What the module produced
        - Feedback: Diagnostic information (errors, reasoning, etc.)
        """
```

**Pre-built Adapters:**
- `DefaultAdapter`: Single-turn LLM system prompt optimization
- `DSPyAdapter`: DSPy module integration
- `GenericRAGAdapter`: RAG system optimization
- `MCPAdapter`: Model Context Protocol tools
- `TerminalBenchAdapter`: Terminal-use agents

---

### 6. Batch Sampling Strategies

**Paper:** Uses minibatch sampling for efficiency

**Code:** `strategies/batch_sampler.py`

```python
class EpochShuffledBatchSampler:
    """Default: shuffle all IDs each epoch, sample sequentially."""

    def next_minibatch_ids(self) -> list[str]:
        if self.position >= len(self.shuffled_ids):
            self._reshuffle()
        batch = self.shuffled_ids[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return batch

class AdaBoostBatchSampler:
    """Weight samples by past performance - focus on harder examples."""

    def next_minibatch_ids(self) -> list[str]:
        weights = self._compute_weights()  # Down-weight easy examples
        return random.choices(self.all_ids, weights=weights, k=self.batch_size)
```

---

### 7. Component Selection Strategies

**Paper:** Round-robin ensures all modules receive updates

**Code:** `strategies/component_selector.py`

```python
class RoundRobinReflectionComponentSelector:
    """Cycle through components sequentially."""

    def select_components(self, candidate: dict) -> list[str]:
        component = self.component_names[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.component_names)
        return [component]

class AllReflectionComponentSelector:
    """Update all components every iteration."""

    def select_components(self, candidate: dict) -> list[str]:
        return list(candidate.keys())
```

---

### 8. Stopping Conditions

**Paper:** Budget-based (rollout count)

**Code:** `utils/` → Various `StopperProtocol` implementations

```python
class MaxMetricCallsStopper:
    """Stop after N metric evaluations."""

class TimeoutStopCondition:
    """Stop after elapsed time."""

class NoImprovementStopper:
    """Stop if no improvement for N iterations."""

class FileStopper:
    """Stop if 'gepa.stop' file exists (graceful shutdown)."""

class CompositeStopper:
    """Combine multiple stoppers (stop if ANY triggers)."""
```

---

## Data Flow Diagram

```
User calls optimize() API
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Initialize GEPAState                                        │
│  - Evaluate seed candidate on validation set                │
│  - Initialize Pareto frontiers                               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Main Loop (GEPAEngine.run)                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. ParetoCandidateSelector.select_candidate_idx()     │  │
│  │ 2. BatchSampler.next_minibatch_ids()                  │  │
│  │ 3. adapter.evaluate(capture_traces=True)              │  │
│  │ 4. ComponentSelector.select_components()              │  │
│  │ 5. adapter.make_reflective_dataset()                  │  │
│  │ 6. LLM reflection → propose new prompt texts          │  │
│  │ 7. adapter.evaluate(capture_traces=False)             │  │
│  │ 8. If score improved:                                 │  │
│  │    - Full validation evaluation                       │  │
│  │    - Add to candidate pool                            │  │
│  │    - Update Pareto frontier                           │  │
│  └───────────────────────────────────────────────────────┘  │
│  Repeat until budget exhausted                               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Return GEPAResult                                           │
│  - best_candidate: Highest aggregate validation score        │
│  - All candidates and their scores                           │
│  - Lineage information                                       │
│  - Optimization statistics                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Sample Efficiency via Two-Stage Evaluation

- **Minibatch evaluation** for proposal acceptance (cheap)
- **Full validation** only after acceptance (expensive)
- Most rollouts go to validation, not learning

### 2. Natural Language as Learning Medium

- Trajectories serialized to text (reasoning, tool calls, errors)
- LLM reflects on text to diagnose problems
- Enables large updates from few examples (vs. small gradient steps)

### 3. Pareto Diversity for Exploration

- Avoids local optima by maintaining multiple "winning" strategies
- Each strategy specializes on different problem instances
- Weighted sampling balances exploration/exploitation

### 4. Lineage Tracking for Merge

- Parent-child relationships enable finding common ancestors
- Merge combines complementary specializations
- Preserves lessons learned along different branches

### 5. Protocol-Based Extensibility

- `GEPAAdapter` protocol for system integration
- Pluggable strategies for selection, sampling, stopping
- Generic types allow adapter-specific data types

---

## References

- Paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
- Authors: Lakshya A Agrawal, Shangyin Tan, et al. (UC Berkeley, Stanford, MIT, etc.)
- Published: July 2025
