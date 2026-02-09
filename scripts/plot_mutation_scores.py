#!/usr/bin/env python3
"""Gather mutation scores across all runs and plot parent score vs. child score."""

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

# Collect all parent/child score pairs
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

            before = row.get("train/batch_score_before")
            after = row.get("train/batch_score_after")

            if before is not None:
                iteration_data[iteration]["before"] = before
            if after is not None:
                iteration_data[iteration]["after"] = after

        # Now collect pairs where we have both before and after
        count = 0
        for iteration, data in iteration_data.items():
            if "before" in data and "after" in data:
                before = float(data["before"])
                after = float(data["after"])
                if not np.isnan(before) and not np.isnan(after):
                    all_data.append({
                        "run_name": run.displayName,
                        "iteration": iteration,
                        "parent_score": before,
                        "child_score": after,
                    })
                    count += 1

        print(f"    Found {count} valid score pairs (from {len(iteration_data)} iterations)")

    except Exception as e:
        print(f"    Error: {e}")
        import traceback

        traceback.print_exc()
        continue

print(f"\nCollected {len(all_data)} mutation score pairs")

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
jitter = 0.15
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
        alpha=0.6,
        s=40,
    )

# Add diagonal line
min_val = min(parent_scores.min(), child_scores.min()) - 0.5
max_val = max(parent_scores.max(), child_scores.max()) + 0.5
ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2)

ax1.set_xlabel("Parent Score (before mutation)", fontsize=12)
ax1.set_ylabel("Child Score (after mutation)", fontsize=12)
ax1.set_title("Scatter Plot (with jitter)", fontsize=14)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)

# === Right plot: 2D histogram (heatmap) ===
ax2 = axes[1]

# Count occurrences at each (parent, child) pair
pair_counts = Counter(zip(parent_scores.astype(int), child_scores.astype(int)))
unique_vals = sorted(set(parent_scores.astype(int)) | set(child_scores.astype(int)))
n_vals = len(unique_vals)
val_to_idx = {v: i for i, v in enumerate(unique_vals)}

# Create count matrix
count_matrix = np.zeros((n_vals, n_vals))
for (p, c), count in pair_counts.items():
    count_matrix[val_to_idx[c], val_to_idx[p]] = count

# Plot heatmap
im = ax2.imshow(count_matrix, origin="lower", cmap="YlOrRd", aspect="equal")
ax2.set_xticks(range(n_vals))
ax2.set_yticks(range(n_vals))
ax2.set_xticklabels(unique_vals)
ax2.set_yticklabels(unique_vals)

# Add count annotations
for i in range(n_vals):
    for j in range(n_vals):
        count = int(count_matrix[i, j])
        if count > 0:
            text_color = "white" if count > count_matrix.max() / 2 else "black"
            ax2.annotate(str(count), (j, i), ha="center", va="center", fontsize=11, color=text_color)

# Add diagonal line
ax2.plot([-0.5, n_vals - 0.5], [-0.5, n_vals - 0.5], "k--", alpha=0.5, linewidth=2)

ax2.set_xlabel("Parent Score (before mutation)", fontsize=12)
ax2.set_ylabel("Child Score (after mutation)", fontsize=12)
ax2.set_title("Frequency Heatmap", fontsize=14)
plt.colorbar(im, ax=ax2, label="Count")

# Save the plot
output_path = "/Users/panwenbo/Repos/gepa/analysis_output/parent_vs_child_score.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")

# Statistics
improvements = child_scores > parent_scores
regressions = child_scores < parent_scores
neutral = child_scores == parent_scores

print(f"\nStatistics:")
print(f"  Total mutations: {len(all_data)}")
print(f"  Improvements (child > parent): {improvements.sum()} ({100*improvements.mean():.1f}%)")
print(f"  Neutral (child == parent): {neutral.sum()} ({100*neutral.mean():.1f}%)")
print(f"  Regressions (child < parent): {regressions.sum()} ({100*regressions.mean():.1f}%)")
print(f"  Mean parent score: {parent_scores.mean():.3f}")
print(f"  Mean child score: {child_scores.mean():.3f}")
print(f"  Mean change: {(child_scores - parent_scores).mean():.3f}")

print("\nBy run:")
for run_name in unique_runs:
    mask = np.array([name == run_name for name in run_names])
    p, c = parent_scores[mask], child_scores[mask]
    n = mask.sum()
    impr = (c > p).sum()
    regr = (c < p).sum()
    print(f"  {run_name}: {n} mutations, {impr} impr ({100*impr/n:.0f}%), {regr} regr ({100*regr/n:.0f}%)")
