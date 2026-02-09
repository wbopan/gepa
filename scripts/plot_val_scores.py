#!/usr/bin/env python3
"""Gather validation scores across all runs and plot parent vs child validation score."""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# Initialize wandb API
api = wandb.Api()

# Entity and project
entity = "bmpixel"
project = "gepa-boost"

# Get all finished runs
runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})

print(f"Found {len(runs)} finished runs")

# Collect all parent/child validation score pairs
all_data = []

for run in runs:
    if run.historyLineCount == 0:
        print(f"  Skipping {run.name} ({run.displayName}): no history")
        continue

    print(f"  Processing {run.name} ({run.displayName}): {run.historyLineCount} history rows")

    try:
        # Fetch all history and aggregate by iteration
        history = run.scan_history()

        # Group by gepa_iteration
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

        # Now collect parent-child pairs
        count = 0
        for iteration, data in sorted(iteration_data.items()):
            if "val/new_score" not in data:
                continue  # No new candidate discovered this iteration

            child_idx = int(data["candidate/new_idx"])
            child_score = float(data["val/new_score"])

            parent_idx = data.get("candidate/selected_idx")
            if parent_idx is None:
                continue  # Seed candidate (no parent)

            parent_idx = int(parent_idx)
            parent_score = candidate_scores.get(parent_idx)

            if parent_score is None:
                print(f"    Warning: No val_score for parent {parent_idx} at iteration {iteration}")
                continue

            all_data.append({
                "run_name": run.displayName,
                "iteration": iteration,
                "parent_idx": parent_idx,
                "child_idx": child_idx,
                "parent_score": parent_score,
                "child_score": child_score,
            })
            count += 1

        print(f"    Found {count} parent-child validation score pairs (from {len(candidate_scores)} candidates)")

    except Exception as e:
        print(f"    Error: {e}")
        import traceback

        traceback.print_exc()
        continue

print(f"\nCollected {len(all_data)} validation score pairs")

if len(all_data) == 0:
    print("No data to plot!")
    exit(1)

# Convert to arrays for plotting
parent_scores = np.array([d["parent_score"] for d in all_data])
child_scores = np.array([d["child_score"] for d in all_data])
run_names = [d["run_name"] for d in all_data]

# Get unique run names for coloring
unique_runs = sorted(set(run_names))

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# === Left plot: Scatter with jitter ===
ax1 = axes[0]
np.random.seed(42)
jitter = 0.01  # Smaller jitter for continuous scores
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runs)))
color_map = {name: colors[i] for i, name in enumerate(unique_runs)}

for run_name in unique_runs:
    mask = np.array([name == run_name for name in run_names])
    n_points = mask.sum()
    jittered_parent = parent_scores[mask] + np.random.uniform(-jitter, jitter, n_points)
    jittered_child = child_scores[mask] + np.random.uniform(-jitter, jitter, n_points)
    ax1.scatter(
        jittered_parent,
        jittered_child,
        c=[color_map[run_name]],
        label=f"{run_name} ({n_points})",
        alpha=0.7,
        s=50,
    )

# Add diagonal line
min_val = min(parent_scores.min(), child_scores.min()) - 0.05
max_val = max(parent_scores.max(), child_scores.max()) + 0.05
ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2, label="y=x (no change)")

ax1.set_xlabel("Parent Validation Score", fontsize=12)
ax1.set_ylabel("Child Validation Score", fontsize=12)
ax1.set_title("Validation Score: Parent vs Child (Scatter)", fontsize=14)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)

# === Right plot: 2D histogram (heatmap) ===
ax2 = axes[1]

# Use hexbin for continuous data
hb = ax2.hexbin(parent_scores, child_scores, gridsize=15, cmap="YlOrRd", mincnt=1)
ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2)

ax2.set_xlabel("Parent Validation Score", fontsize=12)
ax2.set_ylabel("Child Validation Score", fontsize=12)
ax2.set_title("Validation Score: Parent vs Child (Density)", fontsize=14)
ax2.set_aspect("equal")
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)
plt.colorbar(hb, ax=ax2, label="Count")

# Save the plot
output_path = "/Users/panwenbo/Repos/gepa/analysis_output/parent_vs_child_val_score.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")

# Statistics
improvements = child_scores > parent_scores
regressions = child_scores < parent_scores
neutral = child_scores == parent_scores
score_diff = child_scores - parent_scores

print(f"\nStatistics:")
print(f"  Total mutations: {len(all_data)}")
print(f"  Improvements (child > parent): {improvements.sum()} ({100*improvements.mean():.1f}%)")
print(f"  Neutral (child == parent): {neutral.sum()} ({100*neutral.mean():.1f}%)")
print(f"  Regressions (child < parent): {regressions.sum()} ({100*regressions.mean():.1f}%)")
print(f"  Mean parent val score: {parent_scores.mean():.3f}")
print(f"  Mean child val score: {child_scores.mean():.3f}")
print(f"  Mean change: {score_diff.mean():.4f}")
print(f"  Std of change: {score_diff.std():.4f}")

print("\nBy run:")
for run_name in unique_runs:
    mask = np.array([name == run_name for name in run_names])
    p, c = parent_scores[mask], child_scores[mask]
    n = mask.sum()
    if n == 0:
        continue
    impr = (c > p).sum()
    regr = (c < p).sum()
    diff = (c - p).mean()
    print(f"  {run_name}: {n} mutations, {impr} impr ({100*impr/n:.0f}%), {regr} regr ({100*regr/n:.0f}%), avg change: {diff:+.4f}")
