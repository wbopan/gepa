#!/usr/bin/env python3
"""Generate heatmaps showing training sample weights over iterations."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

OUTPUT_DIR = Path("/Users/panwenbo/Repos/gepa/analysis_output")

ENTITY = "bmpixel"
PROJECT = "gepa-boost"
RUNS = {
    "adaboost": "nfadkixt",
    "bayesian": "k0jcelpb",
}


def get_all_weight_tables(run_id: str) -> list[pd.DataFrame]:
    """Get all train_sample_weights tables from all iterations."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    weight_tables = []
    for artifact in run.logged_artifacts():
        if "train_sample_weights" in artifact.name:
            try:
                table_path = artifact.download()
                table_file = os.path.join(table_path, "train_sample_weights.table.json")
                if os.path.exists(table_file):
                    with open(table_file, "r") as f:
                        data = json.load(f)
                        columns = data.get("columns", [])
                        rows = data.get("data", [])
                        df = pd.DataFrame(rows, columns=columns)
                        weight_tables.append(df)
            except Exception as e:
                print(f"Error: {e}")

    return weight_tables


def build_weight_matrix(weight_tables: list[pd.DataFrame]) -> tuple[np.ndarray, list, list]:
    """Build a 2D matrix of weights: rows=samples, cols=iterations."""
    if not weight_tables:
        return None, None, None

    # Find sample and weight columns
    sample_col = None
    weight_col = None
    iteration_col = None

    for col in weight_tables[0].columns:
        if "sample" in col.lower() and "id" in col.lower():
            sample_col = col
        elif col.lower() == "weight" or "weight" in col.lower():
            weight_col = col
        elif "iteration" in col.lower():
            iteration_col = col

    if sample_col is None or weight_col is None:
        print(f"Could not find columns. Available: {list(weight_tables[0].columns)}")
        return None, None, None

    # Get all unique sample IDs
    all_samples = set()
    for df in weight_tables:
        all_samples.update(df[sample_col].unique())

    # Sort samples numerically if possible
    try:
        sample_ids = sorted(all_samples, key=lambda x: int(x))
    except:
        sample_ids = sorted(all_samples)

    sample_to_idx = {s: i for i, s in enumerate(sample_ids)}

    # Build matrix
    n_samples = len(sample_ids)
    n_iterations = len(weight_tables)
    matrix = np.ones((n_samples, n_iterations))  # Default weight = 1

    iterations = []
    for iter_idx, df in enumerate(weight_tables):
        # Get iteration number from the table if available
        if iteration_col and iteration_col in df.columns:
            iter_num = df[iteration_col].iloc[0] if len(df) > 0 else iter_idx
        else:
            iter_num = iter_idx
        iterations.append(iter_num)

        for _, row in df.iterrows():
            sample_id = row[sample_col]
            weight = float(row[weight_col])
            if sample_id in sample_to_idx:
                matrix[sample_to_idx[sample_id], iter_idx] = weight

    return matrix, sample_ids, iterations


def plot_weight_heatmap(matrix: np.ndarray, sample_ids: list, iterations: list,
                        run_name: str, use_log: bool = True):
    """Plot heatmap of weights."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use log scale for better visualization
    if use_log:
        plot_matrix = np.log10(matrix + 0.01)  # Add small value to avoid log(0)
        label = "log₁₀(Weight)"
    else:
        plot_matrix = matrix
        label = "Weight"

    # Create heatmap
    im = ax.imshow(plot_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')

    # Labels
    ax.set_xlabel("Iteration (table index)", fontsize=12)
    ax.set_ylabel("Sample ID", fontsize=12)
    ax.set_title(f"Training Sample Weights Over Time - {run_name.upper()}", fontsize=14)

    # Y-axis ticks (sample IDs)
    ax.set_yticks(range(len(sample_ids)))
    ax.set_yticklabels(sample_ids)

    # X-axis ticks (show every 10th iteration)
    n_iters = len(iterations)
    tick_step = max(1, n_iters // 10)
    ax.set_xticks(range(0, n_iters, tick_step))
    ax.set_xticklabels([str(iterations[i]) for i in range(0, n_iters, tick_step)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=11)

    # Add weight statistics as text
    final_weights = matrix[:, -1]
    max_sample = sample_ids[np.argmax(final_weights)]
    min_sample = sample_ids[np.argmin(final_weights)]

    stats_text = (f"Final iteration stats:\n"
                  f"Max weight: {final_weights.max():.2f} (Sample {max_sample})\n"
                  f"Min weight: {final_weights.min():.3f} (Sample {min_sample})\n"
                  f"Ratio: {final_weights.max()/final_weights.min():.1f}x")

    ax.text(1.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weight_heatmap_{run_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: weight_heatmap_{run_name}.png")


def main():
    for run_name, run_id in RUNS.items():
        print(f"\nProcessing {run_name}...")

        print("  Downloading weight tables...")
        weight_tables = get_all_weight_tables(run_id)
        print(f"  Found {len(weight_tables)} tables")

        print("  Building weight matrix...")
        matrix, sample_ids, iterations = build_weight_matrix(weight_tables)

        if matrix is not None:
            print(f"  Matrix shape: {matrix.shape} (samples x iterations)")
            print("  Generating heatmap...")
            plot_weight_heatmap(matrix, sample_ids, iterations, run_name)
        else:
            print("  Failed to build matrix")

    print("\nDone! Opening heatmaps...")


if __name__ == "__main__":
    main()
