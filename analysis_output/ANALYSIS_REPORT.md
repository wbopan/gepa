# GEPA AdaBoost vs Bayesian Sampler Analysis Report

## Executive Summary

This analysis compares two sampling strategies in the GEPA optimization framework:
1. **AdaBoostBatchSampler** - Weighted sampling based on error rates
2. **BayesianBatchSampler** - Variance-based frontier targeting

Both runs used 30 training samples, 30 validation samples, minibatch size of 5, and ran for ~2000 metric calls.

---

## 1. Candidate Selection and Mutation Analysis

### AdaBoost Run
- **Total candidates generated**: 37
- **Best validation score**: 0.80 (Candidate #5)
- **Best candidate's parent**: Candidate #3 (score: 0.70)

**Parent Candidates that Produced Top 15 Candidates:**
| Parent | Children | Parent Score |
|--------|----------|--------------|
| 5 | 6 | 0.80 |
| 29 | 4 | 0.63 |
| 3 | 1 | 0.70 |
| 17 | 1 | 0.63 |
| 15 | 1 | 0.60 |
| 16 | 1 | 0.73 |
| 0 | 1 | 0.40 |

**Concentration**: Top parent (Candidate #5) produced 40% of top candidates
**Assessment**: Selection is **CONCENTRATED** - one dominant parent produces most successful mutations

### Bayesian Run
- **Total candidates generated**: 30
- **Best validation score**: 0.77 (Candidate #11)
- **Best candidate's parent**: Candidate #7 (score: 0.63)

**Parent Candidates that Produced Top 15 Candidates:**
| Parent | Children | Parent Score |
|--------|----------|--------------|
| 11 | 6 | 0.77 |
| 16 | 3 | 0.67 |
| 7 | 2 | 0.63 |
| 5 | 2 | 0.57 |
| 8 | 1 | 0.67 |
| 2 | 1 | 0.57 |

**Concentration**: Top parent (Candidate #11) produced 40% of top candidates
**Assessment**: Selection is also **CONCENTRATED** - similar pattern to AdaBoost

---

## 2. Training Sample Selection Patterns

### AdaBoost Run

**Sample Selection Frequency (Top 10):**
| Sample | Times Selected | % of Total |
|--------|----------------|------------|
| 1 | 48 | 11.3% |
| 29 | 42 | 9.9% |
| 10 | 41 | 9.6% |
| 17 | 41 | 9.6% |
| 18 | 41 | 9.6% |
| 12 | 39 | 9.2% |
| 8 | 28 | 6.6% |
| 2 | 13 | 3.1% |
| 22 | 13 | 3.1% |
| 5 | 9 | 2.1% |

**Phase Analysis:**
- **Early**: 30 unique samples, balanced selection (most common: 1, 2, 8)
- **Mid**: 23 unique samples, concentration starts (1: 18 times, 17: 16, 18: 16)
- **Late**: 26 unique samples, extreme concentration (1: 21 times, 29: 19, 10: 18)

**Pattern**: Progressive concentration on difficult samples

### Bayesian Run

**Sample Selection Frequency (Top 10):**
| Sample | Times Selected | % of Total |
|--------|----------------|------------|
| 27 | 22 | 4.5% |
| 28 | 22 | 4.5% |
| 24 | 21 | 4.3% |
| 25 | 21 | 4.3% |
| 8 | 19 | 3.9% |
| 10 | 19 | 3.9% |
| 22 | 19 | 3.9% |
| 23 | 19 | 3.9% |
| 6 | 18 | 3.7% |
| 14 | 18 | 3.7% |

**Phase Analysis:**
- **Early**: 30 unique samples, exploration (most common: 16, 24, 10)
- **Mid**: 30 unique samples, still diverse (25, 27, 28)
- **Late**: 30 unique samples, maintains diversity (1, 23, 27)

**Pattern**: Maintains uniform diversity throughout - focuses on "frontier" samples where candidates disagree

---

## 3. Weight Distribution Analysis

### AdaBoost Run

| Phase | Weight Min | Weight Max | Ratio | Gini |
|-------|-----------|-----------|-------|------|
| Early | 0.23 | 4.55 | 20x | 0.53 |
| Mid | 0.04 | 12.26 | 284x | 0.80 |
| Late | 0.02 | 11.14 | 474x | 0.87 |

**Interpretation**:
- Weights become **extremely imbalanced** over time
- Gini coefficient approaches 1.0 (maximum inequality)
- A few difficult samples dominate training

**Consistently High-Weight Samples (Hardest):**
| Sample | Avg Weight | Max Weight |
|--------|------------|------------|
| 1 | 8.12 | 13.22 |
| 18 | 5.79 | 10.26 |
| 17 | 5.38 | 9.46 |
| 29 | 1.77 | 7.09 |
| 10 | 1.73 | 7.56 |

**Consistently Low-Weight Samples (Easiest):**
| Sample | Avg Weight |
|--------|------------|
| 9 | 0.13 |
| 0 | 0.15 |
| 3, 4, 13, 14, 15, 16 | ~0.15 |

### Bayesian Run

| Phase | Weight Min | Weight Max | Ratio | Gini |
|-------|-----------|-----------|-------|------|
| Early | 0.83 | 1.29 | 1.6x | 0.10 |
| Mid | 0.52 | 1.57 | 3.0x | 0.23 |
| Late | 0.39 | 1.77 | 4.5x | 0.30 |

**Interpretation**:
- Weights remain **relatively balanced** throughout
- Gini coefficient stays low (0.1 → 0.3)
- All samples receive attention

**Consistently High-Weight Samples (Frontier/Uncertain):**
| Sample | Avg Weight | Interpretation |
|--------|------------|----------------|
| 25 | 1.50 | High uncertainty |
| 27 | 1.48 | High uncertainty |
| 28 | 1.47 | High uncertainty |
| 24 | 1.39 | High uncertainty |
| 5 | 1.38 | High uncertainty |

**Low-Weight Samples:**
All samples stay within 0.6-0.8 range (no extreme suppression)

---

## 4. Key Findings

### AdaBoost Sampler Characteristics:
1. **Concentrates heavily on hard samples**: Samples 1, 17, 18 receive 8-10x average weight
2. **Extreme weight inequality**: Final Gini = 0.87, ratio 474:1
3. **Some samples nearly abandoned**: Easy samples get weights ~0.02 (almost never selected)
4. **Same samples always selected**: Samples 1, 17, 18, 29, 10 dominate mid/late training
5. **Good for difficult samples**: Forces repeated attempts on hard cases

### Bayesian Sampler Characteristics:
1. **Maintains sample diversity**: All 30 samples selected throughout
2. **Moderate weight spread**: Final Gini = 0.30, ratio 4.5:1
3. **Frontier targeting**: Prioritizes samples where candidates disagree
4. **Sample rotation**: Different samples selected in different phases
5. **Good for exploration**: Ensures all samples receive attention

### Candidate Selection Patterns:
- **Both methods show concentrated parent selection**: The best candidate tends to produce multiple successful offspring
- **AdaBoost produced more candidates** (37 vs 30) with similar best scores (0.80 vs 0.77)

---

## 5. Recommendations

### Use AdaBoost when:
- Some samples are known to be very difficult
- You want to focus compute on hard cases
- Overfitting to easy samples is a concern

### Use Bayesian when:
- Sample difficulty is uncertain
- You want broad coverage of the training set
- Exploration of the sample space is important

---

## Generated Visualizations

1. `candidate_genealogy_adaboost.png` - Parent-child relationships and score distribution
2. `candidate_genealogy_bayesian.png` - Parent-child relationships and score distribution
3. `comparison.png` - Side-by-side comparison of key metrics
4. `weight_distribution_comparison.png` - Weight histograms at early/mid/late phases
5. `weight_evolution_detailed_adaboost.png` - Individual sample weight trajectories
6. `weight_evolution_detailed_bayesian.png` - Individual sample weight trajectories
