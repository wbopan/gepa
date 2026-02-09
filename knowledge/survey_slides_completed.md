#survey

## Recap: GEPA and Prompt Evolution

### 4.1 GEPA: Reflective Prompt Evolution

![[../Attachments/CleanShot 2026-01-26 at 11.46.20@2x.png]]

GEPA is an evolutionary algorithm for optimizing LLM prompts:

- **Iterative Reflective Mutation**: Start with a seed prompt and iteratively propose improvements by sampling minibatches, executing with trace capture, and using an LLM to analyze failures and suggest better text—providing sample-efficient learning through interpretable natural language feedback.

- **Two-Stage Evaluation**: Proposals are first evaluated on cheap minibatches (3-5 examples) to check local improvement; only candidates showing minibatch improvement proceed to full validation, achieving up to 35x fewer rollouts than RL baselines.

- **Pareto-Based Selection**: Rather than greedy optimization, GEPA maintains a Pareto frontier tracking candidates that perform best on different examples, then samples parents proportionally to their "wins"—maintaining diverse strategies that prevent local optima.

- **System-Aware Merge**: When two frontier candidates excel on different subsets, GEPA merges them by selecting the best version of each component, validated only on the intersection of successful examples.

### 4.2 Feedback Descent: Structured Comparison

![[../Attachments/CleanShot 2026-01-26 at 11.47.20@2x.png]]

Feedback Descent extends basic mutation with structured failure analysis:

- **Pairwise Comparison & Failure Analysis**: When a proposed modification fails, a `PairwiseFeedbackGenerator` analyzes parent and child side-by-side, comparing outputs and scores to identify why the change failed in concrete terms.

- **Iterative Loop with History**: The proposer attempts multiple modifications (up to `max_attempts`) on the same minibatch, accumulating failed attempts with their analyses, allowing the LLM to learn from prior mistakes and adjust strategy.

- **Smart Output Truncation**: Intelligent truncation preserves both reasoning (beginning) and final answer (end) of LLM outputs, with per-example feedback showing which items improved or worsened.

- **Early Exit on Success**: The loop terminates when improvement threshold is exceeded, while tracking the best candidate seen throughout iterations even without strict improvement.

---

## 5. Idea: GEPA-Feedback-Descent-Adaboost

### Intuition

- Self-evolving methods work primarily by recording typical patterns and common problems in the prompt
- This includes output formatting, answer preferences, etc.
- These systematic biases are easily optimized—often the reflection LM can fix them after reading 1-2 samples
- What self-evolving doesn't handle well:
    - 1. Samples requiring genuine strong reasoning and complex strategies
    - 2. Diverse sample distributions where learned strategies don't generalize

### 5.1 Key Insight: Borrow from AdaBoost

AdaBoost solves a similar problem:

1. Initialize equal weights for all samples
2. Train weak classifier on weighted samples
3. Increase weights for misclassified samples
4. Repeat to build ensemble focusing on hard cases
5. Combine through weighted voting

**Theoretical guarantee:** Training error decreases exponentially if each classifier beats random guessing.

**Key insight:** Focus each iteration on what the current system does poorly.

### 5.3 The Pareto-Boosting Algorithm

**Step 1: Identify Pareto frontier**

- Find candidates not dominated by any other
- $P_1$ dominates $P_2$ if: $P_1$ ≥ $P_2$ on all samples, strictly better on at least one

**Step 2: Find weak samples for each frontier candidate**

- Identify samples where other frontier candidates outperform it

**Step 3: Optimize on weak samples**

- Generate improved version targeting those weaknesses
- Use Feedback Descent to analyze _why_ it fails

**Step 4: Verify and update**

- Confirm improvements don't catastrophically hurt strong samples
- Add successful improvements to candidate pool

**Step 5: Prune dominated candidates**

- Remove candidates now dominated by others

**Step 6: Iterate until convergence**

### 5.4 AdaBoost Batch Sampler

The AdaBoost batch sampler adaptively focuses training on difficult samples:

**Core Mechanism:**
- Each training sample has a dynamic weight reflecting current difficulty
- Weights update based on evaluation scores using exponential adjustment:
  ```
  new_weight = old_weight × exp(β × (error - 0.5))
  where error = 1.0 - score
  ```

**Weight Behavior:**
| Score | Error | Multiplier | Effect |
|-------|-------|------------|--------|
| 1.0 (perfect) | 0.0 | 0.606 | Weight ↓ 40% |
| 0.5 (neutral) | 0.5 | 1.0 | No change |
| 0.0 (failed) | 1.0 | 1.649 | Weight ↑ 65% |

**Two Variants:**
1. **AdaBoostBatchSampler**: Updates all samples continuously
2. **PMaxBatchSampler** (GEPA default): Three-tier strategy
   - Unattempted samples: Initial boost (weight=1.5)
   - Never-solved samples: AdaBoost updates
   - Once-solved samples: Weight reset to 1.0 (proven solvable, stop boosting)

**Deterministic Selection:** Uses ResidualWeightedSampler (error diffusion) ensuring all samples eventually get selected, with weight=2.0 meaning ~2x selection frequency.

---

## Other Variants

### Train Sampler: PMax and Bayesian

**PMax (Pragmatic Max):**
- **Intuition**: Once a sample is solved once, it's proven achievable—stop over-focusing on it
- **Three-state logic**:
  - Unattempted → high boost (exploration)
  - Attempted but never-solved → AdaBoost (focus on hard)
  - Once-solved → reset to 1.0 (mission accomplished)
- **Best for**: Mixed easy/hard datasets where some samples just need discovery

**Bayesian:**
- **Intuition**: Samples providing strongest signal are those where candidates disagree most
- **Frontier score formula**: `score = 4 × (s+1) × (f+1) / (s+f+2)²`
  - Cold start (0,0) → 1.0 (maximum priority)
  - Balanced (s ≈ f) → 1.0 (maximum disagreement)
  - One-sided (s >> f or f >> s) → low (consistent, less informative)
- **Supports fractional counting**: Partial credit (0.25, 0.5, 0.75) splits between success/failure
- **Best for**: Interactive/uncertain tasks, natural exploration-exploitation balance

| Aspect | AdaBoost | Bayesian | PMax |
|--------|----------|----------|------|
| Primary Goal | Hard samples | Disagreement | Explore & Crack |
| Exploration | Minimal | Automatic | Explicit |
| Score Support | Binary | Continuous | Binary |

### Candidate Selector: Family Avg and Max

**AvgFamilyScoreCandidateSelector:**
- **Formula**: `Family Score = (Parent + Sum of Children) / (1 + Num Children)`
- **Intuition**: Value parents for both their own performance AND average offspring quality
- **Behavior**: Penalizes parents whose children underperform; rewards consistent improvement
- **Weakness**: High-variance parents get punished even with occasional home runs

**MaxFamilyScoreCandidateSelector:**
- **Formula**: `Family Score = max(Parent Score, max(Children Scores))`
- **Selection**: `Weight = (Family Score)^power / sum((Family Scores)^power)` (power=3 default)
- **Intuition**: Value parents by the best achievement in their lineage—don't penalize for variance
- **Behavior**: Captures "this parent found a path to score X" even if other children failed
- **Uses residual weighted sampling** for fairness despite power transform

**Key Finding**: In practice, family-based selectors underperformed Pareto-based selection. Analysis shows:
- Avg family got stuck on mediocre parents (0.471 avg) before escaping
- Max family with power=3 over-exploited, creating a "gravity well" around mediocre candidates

### Task LM and Datasets

To verify generalizability, we tested across different datasets and models:

- Reflection LM: Deepseek V3.2
- GPQA-diamond with Qwen-235b
- GPQA-diamond with GPT5-mini
- NYT-connections with Deepseek V3.2

---

## Questions to Answer

We have several questions to answer:

1. Does AdaBoost or Bayesian minibatch sampling work?
2. By focusing on hard training samples, are we getting good valset performance?
3. Is random sampling on frontier candidate needed or can we use argmax?
4. How reproducible are evolution runs?
5. How do different models react to GEPA?
6. How do different datasets react to GEPA?

### Q1: Does AdaBoost or Bayesian Minibatch Sampling Work?

![[../analysis_output/q1_sampler_comparison.png]]

**Conclusion: Works slightly better, but basically equally well.**

| Sampler | Best Score | Pareto Agg | Candidates |
|---------|-----------|------------|------------|
| Baseline (Random) | 0.77 | 0.90 | 38 |
| AdaBoost | **0.80** | 0.93 | 37 |
| Bayesian | 0.77 | **0.97** | 30 |
| PMax | 0.67 | 0.83 | 8 |

**Key observations:**
- AdaBoost achieves highest single-candidate score (0.80)
- Bayesian achieves highest ensemble/oracle score (0.97)
- PMax underperforms due to limited exploration (only 8 candidates)
- All adaptive methods show marginal improvement over random baseline

### Q2: Hard Training Samples → Valset Performance?

![[../analysis_output/weight_heatmap_adaboost.png]]

**Training weight evolution shows clear patterns:**
- Persistently hard samples (IDs 1, 12, 17, 18) maintain high weights throughout
- Successfully solved samples (IDs 13, 15, 22) see weight reduction
- Weight ratio reaches 473.8x between hardest and easiest samples

**Correlation analysis (170 parent-child pairs):**

| Parent Score | Avg Child Score | Improvement | Success Rate |
|-------------|-----------------|-------------|--------------|
| 0.8-1.0 | 0.650 | -0.150 | 0% |
| 0.6-0.7 | 0.622 | -0.015 | 38.5% |
| 0.4-0.5 | 0.389 | -0.025 | 40% |

**Conclusion:** Focusing on hard samples helps exploration but doesn't directly translate to valset improvement. Mean reversion dominates.

### Q3: Random Sampling vs Argmax on Frontier Candidate?

![[../analysis_output/q3_selector_comparison.png]]

**Conclusion: Random (Pareto-based) selection outperforms deterministic selectors.**

| Selector | Best Score | Avg Parent Score |
|----------|-----------|------------------|
| Pareto (Default) | **0.80** | 0.715 |
| Family Avg | 0.53 | 0.471 |
| Family Max (power=1) | 0.73 | 0.472 |
| Family Max (power=3) | 0.63 | 0.533 |

**Root cause of family selector failure:**
- Power=3 creates "gravity well" around mediocre candidates
- 71% of max3 mutations concentrated on 2 mediocre lineages
- Fertility cliff: Parents selected 5+ times never produced improvements

### Q4: How Reproducible Are Evolution Runs?

![[../analysis_output/q4_reproducibility.png]]

**Comparing adaboost-235b vs adaboost-rerun-235b:**

| Metric | Original Run | Rerun | Difference |
|--------|-------------|-------|------------|
| Best Score | 0.800 | 0.767 | -0.033 |
| Pareto Agg | 0.933 | 0.867 | -0.066 |
| Iterations | 92 | 36 | -56 |
| Candidates | 37 | 10 | -27 |

*Data verified from wandb runs: adaboost-235b (nfadkixt), adaboost-rerun-235b (tefzfm25)*

**Key findings:**
- Best scores differ by ~0.03 (within noise for n=30 valset)
- Rerun was cut short (36 vs 92 iterations), limiting fair comparison
- Early lucky discoveries dominate final results

**Reproducibility concerns:**
- High variance in early phase determines trajectory
- Different runs may find different local optima
- Longer runs tend to find better candidates (more exploration)
- Ensemble (Pareto aggregate) is more stable than single-best

### Q5: How Do Different Models React to GEPA?

![[../analysis_output/q5_model_comparison.png]]

**Qwen-235B vs GPT5-mini on GPQA-Diamond:**

| Model | Baseline | AdaBoost | Pareto Agg | Improvement |
|-------|----------|----------|------------|-------------|
| Qwen-235B | 0.767 | 0.800 | 0.933 | +4.3% |
| GPT5-mini | 0.800 | 0.800 | 0.867 | 0% |

*Data verified from wandb runs: baseline-235b, adaboost-235b, baseline-5mini, adaboost-5mini*

**Key insights:**
- GPT5-mini achieved same best score (0.80) as Qwen-235B with AdaBoost
- GPT5-mini baseline already strong (0.80), leaving no room for single-candidate improvement
- Qwen-235B shows modest improvement (+4.3%) from evolution
- Pareto aggregate (ensemble) shows Qwen-235B benefits more from diversity (0.933 vs 0.867)

### Q6: How Do Different Datasets React to GEPA?

![[../analysis_output/q6_dataset_comparison.png]]

**GPQA-Diamond vs NYT-Connections:**

| Dataset | Seed Score | Best Single | Pareto Agg | Improvement |
|---------|------------|-------------|------------|-------------|
| GPQA-Diamond (Qwen-235B) | 0.400 | 0.800 | 0.933 | +100% |
| NYT-Connections (AdaBoost) | 0.333 | 0.425 | 0.900 | +27.6% |

*Data verified from wandb runs: adaboost-235b, nyt-connections-adaboost*

**Key insights:**
- Both datasets show significant improvement from seed to best candidate
- GPQA-Diamond achieves higher absolute scores but starts from higher seed
- NYT-Connections Pareto aggregate (0.90) suggests strong ensemble potential
- Ensemble/oracle scores much higher than single-best for both datasets

---

## The Most Important Insight

![[../analysis_output/mean_reversion_insight.png]]

At least for all scenarios we've tested, the biggest issue is that GEPA (or similar prompt evolution methods) is largely **mean-reverting evolution**:

### Key Evidence

**1. Child scores don't improve over generations:**

| Parent Score | N Pairs | Avg Child Score | Avg Improvement | Success Rate |
|-------------|---------|-----------------|-----------------|--------------|
| **0.8-1.0** | 18 | 0.650 | **-0.150** | **0%** |
| 0.7-0.8 | 35 | 0.609 | -0.131 | 5.7% |
| 0.6-0.7 | 26 | 0.622 | -0.015 | 38.5% |
| 0.5-0.6 | 40 | 0.453 | -0.087 | 25.0% |
| 0.4-0.5 | 50 | 0.389 | -0.025 | 40.0% |
| **0-0.4** | 1 | 0.567 | **+0.200** | **100%** |

**2. High-scoring parents don't produce high-scoring children:**
- Overall improvement rate: 25.3% (75% of mutations worsen performance)
- Overall average improvement: -0.072
- Correlation between parent score and improvement: r ≈ -0.5

**3. Success comes from exploration, not exploitation:**
- Best improvement: +0.300 (seed 0.4 → child 0.7)
- Worst regression: -0.467 (from parent 0.533)
- High-scoring parents (0.8+) have 61% regression rate

### Conclusion

The effectiveness of GEPA largely comes from:
- Testing various prompt variants (which don't differ much fundamentally)
- Getting lucky on the validation set
- **NOT** from systematically building on high-scoring candidates

This suggests prompt evolution methods may have fundamental limitations for tasks requiring genuine capability improvements rather than surface-level prompt tuning.

---

## Summary of Findings

| Question | Finding |
|----------|---------|
| AdaBoost vs Bayesian | Both work slightly better than random, ~equal to each other |
| Hard samples → valset | Weak correlation; mean reversion dominates |
| Random vs argmax selection | Random (Pareto) wins; family selectors trap in local optima |
| Reproducibility | Moderate variance; rerun shows 0.03 best-score difference |
| Model comparison | Both models reach 0.80 best; 235B has better ensemble (0.93 vs 0.87) |
| Dataset comparison | GPQA: 0.80 best, NYT: 0.425 best; both have high Pareto agg (~0.90) |
| **Most important** | Evolution is mean-reverting; high parents don't produce high children |

*All data verified against wandb (entity: bmpixel, project: gepa-boost)*
