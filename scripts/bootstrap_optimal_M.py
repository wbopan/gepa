#!/usr/bin/env python3
"""Bootstrap simulation to find optimal minibatch size M.

Given:
- N: Total evaluation budget
- K: Validation set size (30)
- M: Minibatch size (to optimize)

We simulate the gate behavior at different M values using our empirical data.
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy import stats

# Initialize wandb API
api = wandb.Api()
entity = "bmpixel"
project = "gepa-boost"

print("Loading data from wandb...")

# Get all finished runs
runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})

# Collect all parent/child validation score pairs (same as before)
all_data = []

for run in runs:
    if run.historyLineCount == 0:
        continue

    try:
        history = run.scan_history()
        iteration_data = defaultdict(dict)
        for row in history:
            iteration = row.get("gepa_iteration")
            if iteration is None:
                continue
            for key in ["val/new_score", "candidate/new_idx", "candidate/selected_idx"]:
                val = row.get(key)
                if val is not None:
                    iteration_data[iteration][key] = val

        # Build candidate_idx -> val_score mapping
        candidate_scores = {}
        for iteration, data in sorted(iteration_data.items()):
            if "val/new_score" in data and "candidate/new_idx" in data:
                idx = int(data["candidate/new_idx"])
                score = float(data["val/new_score"])
                candidate_scores[idx] = score

        # Collect parent-child pairs
        for iteration, data in sorted(iteration_data.items()):
            if "val/new_score" not in data:
                continue
            child_idx = int(data["candidate/new_idx"])
            child_score = float(data["val/new_score"])
            parent_idx = data.get("candidate/selected_idx")
            if parent_idx is None:
                continue
            parent_idx = int(parent_idx)
            parent_score = candidate_scores.get(parent_idx)
            if parent_score is None:
                continue

            all_data.append({
                "run_name": run.displayName,
                "parent_score": parent_score,
                "child_score": child_score,
                "true_delta": child_score - parent_score,
            })

    except Exception as e:
        continue

print(f"Loaded {len(all_data)} parent-child pairs")

# Convert to arrays
true_deltas = np.array([d["true_delta"] for d in all_data])
parent_scores = np.array([d["parent_score"] for d in all_data])
child_scores = np.array([d["child_score"] for d in all_data])

# Parameters
K = 30  # Validation set size
N_SIMULATIONS = 1000  # Monte Carlo simulations per M value
M_VALUES = list(range(1, 31))  # Test M from 1 to 30

# Estimate noise variance for minibatch scores
# For binary outcomes, variance of difference ≈ (p1(1-p1) + p2(1-p2)) / M
# We estimate this from the average scores
avg_score = (parent_scores.mean() + child_scores.mean()) / 2
# Variance of single binary outcome with prob p: p(1-p)
# Variance of difference of two means: 2 * p(1-p) / M (approximately)
base_variance = 2 * avg_score * (1 - avg_score)
print(f"Average score: {avg_score:.3f}")
print(f"Estimated base variance (for M=1): {base_variance:.3f}")


def simulate_gate(true_delta, M, base_var, n_sims=1000):
    """Simulate minibatch gate for a single mutation.

    Returns: probability of passing gate (observed_delta > 0)
    """
    # Noise std for minibatch of size M
    noise_std = np.sqrt(base_var / M)

    # Simulate observed deltas
    observed_deltas = true_delta + np.random.normal(0, noise_std, n_sims)

    # Gate passes if observed delta > 0
    pass_rate = (observed_deltas > 0).mean()
    return pass_rate


def run_simulation(M, true_deltas, base_var, K, n_sims=1000):
    """Run full simulation for a given M value.

    Returns dict with:
    - gate_pass_rate: Overall probability of passing gate
    - true_positive_rate: P(true improvement | passed gate)
    - expected_gain: Expected true delta given passed gate
    - cost_per_iteration: Expected cost (2M for gate + K*pass_rate for validation)
    - efficiency: Expected gain / cost
    """
    np.random.seed(42)  # Reproducibility

    n_mutations = len(true_deltas)

    total_passed = 0
    total_true_positive = 0
    total_gain_if_passed = 0

    for true_delta in true_deltas:
        # Simulate pass rate for this mutation
        noise_std = np.sqrt(base_var / M)
        observed_deltas = true_delta + np.random.normal(0, noise_std, n_sims)
        passes = observed_deltas > 0

        pass_rate = passes.mean()
        total_passed += pass_rate

        # True positive: passed AND actually improved
        if true_delta > 0:
            total_true_positive += pass_rate

        # Expected gain if passed (true delta * probability of passing)
        total_gain_if_passed += true_delta * pass_rate

    gate_pass_rate = total_passed / n_mutations

    # True positive rate (precision): P(true improvement | passed)
    # = (# true improvements that passed) / (# total passed)
    n_true_improvements = (true_deltas > 0).sum()
    if total_passed > 0:
        precision = total_true_positive / total_passed
    else:
        precision = 0

    # Expected gain per mutation (averaged over all mutations)
    expected_gain = total_gain_if_passed / n_mutations

    # Cost model:
    # - Gate evaluation: 2*M (evaluate both parent and child on M samples)
    # - Full validation: K (only if passed gate)
    cost_gate = 2 * M
    cost_validation = gate_pass_rate * K
    cost_per_iteration = cost_gate + cost_validation

    # Efficiency: expected gain per unit cost
    efficiency = expected_gain / cost_per_iteration if cost_per_iteration > 0 else 0

    return {
        "M": M,
        "gate_pass_rate": gate_pass_rate,
        "precision": precision,  # P(true improvement | passed)
        "expected_gain": expected_gain,
        "cost_gate": cost_gate,
        "cost_validation": cost_validation,
        "cost_total": cost_per_iteration,
        "efficiency": efficiency,
    }


print("\nRunning simulations for different M values...")
results = []
for M in M_VALUES:
    result = run_simulation(M, true_deltas, base_variance, K, N_SIMULATIONS)
    results.append(result)
    if M <= 10 or M % 5 == 0:
        print(f"  M={M:2d}: pass_rate={result['gate_pass_rate']:.2f}, "
              f"precision={result['precision']:.2f}, "
              f"efficiency={result['efficiency']:.4f}")

# Find optimal M
efficiencies = [r["efficiency"] for r in results]
optimal_idx = np.argmax(efficiencies)
optimal_M = results[optimal_idx]["M"]
print(f"\n*** Optimal M* = {optimal_M} ***")
print(f"    Gate pass rate: {results[optimal_idx]['gate_pass_rate']:.2f}")
print(f"    Precision: {results[optimal_idx]['precision']:.2f}")
print(f"    Efficiency: {results[optimal_idx]['efficiency']:.4f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

Ms = [r["M"] for r in results]

# Plot 1: Gate Pass Rate vs M
ax1 = axes[0, 0]
ax1.plot(Ms, [r["gate_pass_rate"] for r in results], "b-o", markersize=4)
ax1.axhline(y=(true_deltas > 0).mean(), color="r", linestyle="--",
            label=f"True improvement rate ({(true_deltas > 0).mean():.2f})")
ax1.axvline(x=optimal_M, color="g", linestyle=":", alpha=0.7)
ax1.set_xlabel("Minibatch Size M")
ax1.set_ylabel("Gate Pass Rate")
ax1.set_title("Gate Pass Rate vs M")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Precision vs M
ax2 = axes[0, 1]
ax2.plot(Ms, [r["precision"] for r in results], "b-o", markersize=4)
ax2.axvline(x=optimal_M, color="g", linestyle=":", alpha=0.7)
ax2.set_xlabel("Minibatch Size M")
ax2.set_ylabel("Precision P(true impr | passed)")
ax2.set_title("Precision vs M")
ax2.grid(True, alpha=0.3)

# Plot 3: Cost breakdown vs M
ax3 = axes[0, 2]
ax3.plot(Ms, [r["cost_gate"] for r in results], "b-o", markersize=4, label="Gate cost (2M)")
ax3.plot(Ms, [r["cost_validation"] for r in results], "r-s", markersize=4, label="Validation cost (K × pass_rate)")
ax3.plot(Ms, [r["cost_total"] for r in results], "k-^", markersize=4, label="Total cost")
ax3.axvline(x=optimal_M, color="g", linestyle=":", alpha=0.7)
ax3.set_xlabel("Minibatch Size M")
ax3.set_ylabel("Cost (# evaluations)")
ax3.set_title("Cost Breakdown vs M")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Expected Gain vs M
ax4 = axes[1, 0]
ax4.plot(Ms, [r["expected_gain"] for r in results], "b-o", markersize=4)
ax4.axvline(x=optimal_M, color="g", linestyle=":", alpha=0.7)
ax4.set_xlabel("Minibatch Size M")
ax4.set_ylabel("Expected Gain")
ax4.set_title("Expected Gain per Mutation vs M")
ax4.grid(True, alpha=0.3)

# Plot 5: Efficiency vs M (THE KEY PLOT)
ax5 = axes[1, 1]
ax5.plot(Ms, efficiencies, "b-o", markersize=4)
ax5.axvline(x=optimal_M, color="g", linestyle=":", alpha=0.7, label=f"Optimal M*={optimal_M}")
ax5.scatter([optimal_M], [efficiencies[optimal_idx]], color="red", s=100, zorder=5)
ax5.set_xlabel("Minibatch Size M")
ax5.set_ylabel("Efficiency (Gain / Cost)")
ax5.set_title("*** EFFICIENCY vs M ***")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Summary text
ax6 = axes[1, 2]
ax6.axis("off")
summary_text = f"""
BOOTSTRAP SIMULATION RESULTS
============================

Data: {len(true_deltas)} mutations
Validation set size K = {K}

True improvement rate: {(true_deltas > 0).mean():.1%}
True mean Δ: {true_deltas.mean():.4f}

Current M ≈ 5:
  Gate pass rate: {results[4]['gate_pass_rate']:.1%}
  Precision: {results[4]['precision']:.1%}
  Efficiency: {results[4]['efficiency']:.4f}

OPTIMAL M* = {optimal_M}:
  Gate pass rate: {results[optimal_idx]['gate_pass_rate']:.1%}
  Precision: {results[optimal_idx]['precision']:.1%}
  Efficiency: {results[optimal_idx]['efficiency']:.4f}

Improvement: {(efficiencies[optimal_idx] / efficiencies[4] - 1) * 100:.1f}%
over current M=5

Recommendation:
{'Increase M to ' + str(optimal_M) if optimal_M > 5 else 'Current M is near optimal'}
"""
ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment="center", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
output_path = "/Users/panwenbo/Repos/gepa/analysis_output/optimal_minibatch_size.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")

# Additional analysis: sensitivity around optimal
print("\nDetailed results around optimal M:")
print("-" * 70)
print(f"{'M':>3} | {'Pass Rate':>10} | {'Precision':>10} | {'Cost':>8} | {'Efficiency':>10}")
print("-" * 70)
for r in results:
    marker = " ***" if r["M"] == optimal_M else ""
    print(f"{r['M']:>3} | {r['gate_pass_rate']:>10.2%} | {r['precision']:>10.2%} | "
          f"{r['cost_total']:>8.1f} | {r['efficiency']:>10.5f}{marker}")
