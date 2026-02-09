# AdaBoost-Max3 vs AdaBoost-Max Performance Analysis

> Analysis Date: 2026-02-02
> WandB Project: bmpixel/gepa-boost

## 1. Experiment Overview

### Run Information
| Run | WandB ID | Git Commit | Key Change |
|-----|----------|------------|------------|
| adaboost-max | s0bupke3 | ac6a8d5 | Original MaxFamilyScoreCandidateSelector (linear weights) |
| adaboost-max3 | znn618br | 9c5b041 | Added power=3 transform + normalization |

### Code Difference

**Original (adaboost-max)**:
```python
# Uses raw family scores directly as weights
self._sampler.update_weights(family_scores)  # e.g., [0.8, 0.6, 0.4]
```

**New (adaboost-max3)**:
```python
# Power transform + normalization
raw_weights = [score ** 3 for score in family_scores]  # [0.512, 0.216, 0.064]
weights = [w / sum(raw_weights) for w in raw_weights]  # Normalized to sum=1.0
```

**Selection Ratio Impact**:
- Linear (original): 0.8 vs 0.4 = **2x** selection ratio
- Power=3: 0.8 vs 0.4 = **8x** selection ratio

---

## 2. Final Performance Comparison

| Metric | adaboost-max | adaboost-max3 |
|--------|--------------|---------------|
| Iterations | 25 | 91 |
| Total Candidates | 13 | 36 |
| **Best Score** | **0.733** | **0.633** |
| val/best_agg | 0.733 | 0.633 |
| pareto/val_agg | 0.867 | 0.933 |
| Runtime | ~1 hour | ~10 hours |

**Key Observation**: Despite running 3.6x longer and generating 2.8x more candidates, adaboost-max3 achieved a **lower** best score.

---

## 3. Candidate Evolution Data

### adaboost-max (13 candidates)
```
idx=0, parent=None, score=0.400  (seed)
idx=1, parent=0, score=0.533, improvement=+0.133
idx=2, parent=0, score=0.133, improvement=-0.267
idx=3, parent=0, score=0.533, improvement=+0.133
idx=4, parent=3, score=0.533, improvement=+0.000
idx=5, parent=0, score=0.700, improvement=+0.300  ← KEY DISCOVERY
idx=6, parent=0, score=0.133, improvement=-0.267
idx=7, parent=0, score=0.533, improvement=+0.133
idx=8, parent=5, score=0.633, improvement=-0.067
idx=9, parent=5, score=0.733, improvement=+0.033  ← BEST
idx=10, parent=3, score=0.600, improvement=+0.067
idx=11, parent=0, score=0.133, improvement=-0.267
idx=12, parent=0, score=0.333, improvement=-0.067
```

### adaboost-max3 (36 candidates)
```
idx=0, parent=None, score=0.400  (seed)
idx=1, parent=0, score=0.533, improvement=+0.133
idx=2, parent=0, score=0.133, improvement=-0.267
idx=3, parent=0, score=0.133, improvement=-0.267
idx=4, parent=0, score=0.067, improvement=-0.333
idx=5, parent=1, score=0.533, improvement=+0.000
idx=6, parent=1, score=0.533, improvement=+0.000
idx=7, parent=0, score=0.633, improvement=+0.233  ← BEST (never exceeded!)
idx=8, parent=5, score=0.533, improvement=+0.000
idx=9, parent=0, score=0.600, improvement=+0.200
idx=10, parent=7, score=0.633, improvement=+0.000
idx=11, parent=1, score=0.300, improvement=-0.233
idx=12, parent=5, score=0.467, improvement=-0.067
idx=13, parent=9, score=0.600, improvement=+0.000
idx=14, parent=10, score=0.500, improvement=-0.133
idx=15, parent=6, score=0.600, improvement=+0.067
idx=16, parent=9, score=0.533, improvement=-0.067
idx=17, parent=1, score=0.200, improvement=-0.333
idx=18, parent=0, score=0.167, improvement=-0.233
idx=19, parent=10, score=0.533, improvement=-0.100
idx=20, parent=14, score=0.600, improvement=+0.100
idx=21, parent=0, score=0.400, improvement=+0.000
idx=22, parent=1, score=0.433, improvement=-0.100
idx=23, parent=5, score=0.500, improvement=-0.033
idx=24, parent=8, score=0.567, improvement=+0.033
idx=25, parent=0, score=0.433, improvement=+0.033
idx=26, parent=10, score=0.500, improvement=-0.133
idx=27, parent=14, score=0.467, improvement=-0.033
idx=28, parent=13, score=0.167, improvement=-0.433
idx=29, parent=1, score=0.533, improvement=+0.000
idx=30, parent=7, score=0.533, improvement=-0.100
idx=31, parent=24, score=0.367, improvement=-0.200
idx=32, parent=12, score=0.333, improvement=-0.133
idx=33, parent=14, score=0.567, improvement=+0.067
idx=34, parent=23, score=0.467, improvement=-0.033
idx=35, parent=7, score=0.600, improvement=-0.033
```

---

## 4. Early Mutation Quality Comparison

### First Good Candidate Discovery
| Run | First ≥0.6 Candidate | Score | Discovery Step |
|-----|---------------------|-------|----------------|
| adaboost-max | idx=5 | **0.700** | 230 |
| adaboost-max3 | idx=7 | **0.633** | 460 |

### Early Mutations from Seed (Parent=0)
**adaboost-max**: +0.133, -0.267, +0.133, **+0.300**, -0.267, +0.133
**adaboost-max3**: +0.133, -0.267, -0.267, -0.333, **+0.233**, +0.200

**Key Observation**: adaboost-max's early mutation produced a +0.300 jump (idx=5), while adaboost-max3's best early jump was only +0.233 (idx=7).

---

## 5. Parent Selection Patterns

### adaboost-max (26 selections)
```
Parent selection counts: {0: 14, 1: 4, 3: 4, 4: 2, 5: 2}
```

### adaboost-max3 (92 selections)
```
Parent selection counts: {0: 20, 1: 14, 5: 8, 7: 8, 6: 7, 10: 6, 9: 5, 8: 4, 13: 4, 15: 3, 14: 3, 12: 2, 16: 2, 19: 2, 20: 2, 24: 1, 23: 1}
```

---

## 6. Parent Fertility Analysis

### adaboost-max
```
Parent 0 (score=0.400): 8 children, avg_improvement=-0.021, best_child=0.700
Parent 3 (score=0.533): 2 children, avg_improvement=+0.033, best_child=0.600
Parent 5 (score=0.700): 2 children, avg_improvement=-0.017, best_child=0.733

High-score parents (>=0.5): avg_improvement=+0.008
Low-score parents (<0.5): avg_improvement=-0.021
```

### adaboost-max3
```
Parent 0 (score=0.400): 9 children, avg_improvement=-0.056, best_child=0.633
Parent 1 (score=0.533): 6 children, avg_improvement=-0.111, best_child=0.533
Parent 5 (score=0.533): 3 children, avg_improvement=-0.033, best_child=0.533
Parent 7 (score=0.633): 3 children, avg_improvement=-0.044, best_child=0.633
Parent 10 (score=0.633): 3 children, avg_improvement=-0.122, best_child=0.533
Parent 14 (score=0.500): 3 children, avg_improvement=+0.044, best_child=0.600
Parent 9 (score=0.600): 2 children, avg_improvement=-0.033, best_child=0.600
Parent 6 (score=0.533): 1 children, avg_improvement=+0.067, best_child=0.600
Parent 8 (score=0.533): 1 children, avg_improvement=+0.033, best_child=0.567
Parent 12 (score=0.467): 1 children, avg_improvement=-0.133, best_child=0.333

High-score parents (>=0.5): avg_improvement=-0.079
Low-score parents (<0.5): avg_improvement=-0.094
```

---

## 7. Later Mutation Analysis (After First Good Candidate)

| Run | Mutations after first good | Improvements over first good | Avg parent score |
|-----|---------------------------|------------------------------|------------------|
| adaboost-max | 7 | **1** (idx=9: 0.733) | 0.505 |
| adaboost-max3 | 28 | **0** | 0.537 |

### Parent Score Distribution for Later Mutations
| Run | High (≥0.6) | Mid (0.4-0.6) | Low (<0.4) |
|-----|-------------|---------------|------------|
| adaboost-max | 2 | 5 | 0 |
| adaboost-max3 | 9 | 19 | 0 |

**Key Finding**: adaboost-max3 had 4.5x more high-score parent selections but **zero** improvements over its best score.

---

## 8. Lineage Diversity Analysis

### adaboost-max (First-Generation Lineages)
```
Lineage from idx=1 (score=0.533): 1 descendants, best=0.533
Lineage from idx=2 (score=0.133): 1 descendants, best=0.133
Lineage from idx=3 (score=0.533): 3 descendants, best=0.600
Lineage from idx=5 (score=0.700): 3 descendants, best=0.733  ← WINNER
Lineage from idx=6 (score=0.133): 1 descendants, best=0.133
Lineage from idx=7 (score=0.533): 1 descendants, best=0.533
Lineage from idx=11 (score=0.133): 1 descendants, best=0.133
Lineage from idx=12 (score=0.333): 1 descendants, best=0.333
```

### adaboost-max3 (First-Generation Lineages)
```
Lineage from idx=1 (score=0.533): 15 descendants, best=0.600  ← OVER-EXPLOITED
Lineage from idx=2 (score=0.133): 1 descendants, best=0.133
Lineage from idx=3 (score=0.133): 1 descendants, best=0.133
Lineage from idx=4 (score=0.067): 1 descendants, best=0.067
Lineage from idx=7 (score=0.633): 10 descendants, best=0.633  ← STUCK AT PARENT SCORE
Lineage from idx=9 (score=0.600): 4 descendants, best=0.600
Lineage from idx=18 (score=0.167): 1 descendants, best=0.167
Lineage from idx=21 (score=0.400): 1 descendants, best=0.400
Lineage from idx=25 (score=0.433): 1 descendants, best=0.433
```

**Key Finding**: adaboost-max3 concentrated 25/35 mutations (71%) on just two lineages (idx=1 and idx=7), neither of which could exceed their first-generation score.

---

## 9. Prompt Content Comparison

### Best Candidate Structural Differences
| Keyword | adaboost-max (0.700) | adaboost-max3 (0.633) |
|---------|---------------------|----------------------|
| Maxwell equations | ✓ | ✗ |
| Harmonic oscillator | ✓ | ✗ |
| Epistasis (genetics) | ✓ | ✗ |
| Key requirements section | ✓ | ✗ |
| Pitfalls section | ✓ | ✗ |
| Step-by-step | ✓ | ✓ |
| Spin | ✓ | ✓ |

**Key Finding**: adaboost-max's best candidate contained critical domain knowledge that adaboost-max3's best candidate never discovered.

---

## 10. Conclusion

### Primary Cause: Bad Luck in Early Mutations
- adaboost-max discovered a high-quality candidate (idx=5, 0.700) with +0.300 improvement
- adaboost-max3's best early candidate (idx=7, 0.633) was only +0.233 improvement
- The 0.700 candidate contained critical domain knowledge (Maxwell, epistasis, etc.) that 0.633 lacked

### Secondary Cause: Power^3 Over-Exploitation
- Power^3 caused 71% of mutations to concentrate on two mediocre lineages
- Despite 28 later mutations, none exceeded the 0.633 local optimum
- The original linear weighting allowed more balanced exploration

### Contribution Assessment
| Factor | Contribution Level |
|--------|-------------------|
| Bad luck in early mutations | **Primary** |
| Power^3 over-exploitation | **Secondary/Aggravating** |

### Key Statistical Evidence
1. adaboost-max3 had **4x more mutations** but found **no improvements** after first good candidate
2. adaboost-max3 selected from high-score parents **4.5x more often** but still failed
3. Both lineages idx=1 and idx=7 were stuck at their first-generation scores despite heavy exploitation

---

## 11. Gemini Debate Summary (2026-02-02)

### Gemini's Counter-Argument

Gemini argued that I **underweighted the systematic failure** and **overweighted bad luck**:

1. **Parent 0 was selected MORE in max3** (20 times vs 14 times), yet max3 still didn't find the 0.700 candidate. This suggests the mutation is highly stochastic.

2. **Fisher's Exact Test**: Comparing 1/8 success (max) vs 0/9 success (max3) gives p-value ≈ 0.47 - **statistically indistinguishable**. Both runs had similar "luck" with parent 0.

3. **The real failure**: Once max3 found mediocre candidates (idx=1 at 0.533, idx=7 at 0.633), power^3 created a "gravity well" that sucked in all resources. Linear weighting in max correctly ignored idx=1 (only 1 descendant).

4. **Power^3 is intrinsically problematic** for early-stage evolution because it turns GA into a "Greedy Hill Climber" that can't escape local optima.

### Revised Causal Attribution

| Factor | Claude's Original | Gemini's Revision |
|--------|-------------------|-------------------|
| Bad luck in early mutations | Primary | Initiating/Secondary |
| Power^3 over-exploitation | Secondary | **Primary** |

**Gemini's Key Insight**: A robust algorithm survives bad luck by maintaining diversity. Power^3 collapses into local optima when early mutations are unlucky.

### Suggested Experiments (Budget: 2 Runs)

**Run 1: "Golden Seed" Test**
- Config: adaboost-max3 (power=3)
- Intervention: Start with seed + the 0.700 candidate from max
- Hypothesis: If power^3 still stagnates, it's systematically flawed

**Run 2: "Consistency" Test**
- Config: adaboost-max (linear)
- Intervention: None, just re-run
- Hypothesis: If linear achieves ≥0.700 again, it's robust

### Practical Recommendations (Ranked)

| Option | Safety | Upside | Complexity |
|--------|--------|--------|------------|
| **Rank-Based Selection** | Highest | High | Low |
| Power=1.5 | High | Medium | Lowest |
| Revert to Linear | High | Low | Low |
| Dynamic Scheduling | Medium | High | Medium |
| Lineage Penalty | Low | High | High |

**Gemini's Top Pick**: Rank-Based Selection - decouples selection pressure from score magnitude, preventing mediocre candidates from creating "gravity wells".

**One-Line Fix**: Change power from 3 to 1.5.

---

## 12. Final Recommendations

1. **Immediate fix**: Change power=3 to power=1.5 (one-line change)
2. **Better fix**: Implement rank-based selection to decouple selection pressure from score magnitude
3. **Validate**: Run the "Golden Seed" and "Consistency" tests to confirm diagnosis
4. **Long-term**: Consider lineage diversity bonus + semantic similarity penalty
