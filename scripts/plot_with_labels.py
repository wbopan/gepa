"""Plot parent-child validation scores with child_idx labels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mutation_data(json_path: str) -> list[dict]:
    """Load mutation analysis data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def plot_valset_scatter_with_labels(
    pairs: list[dict],
    output_path: str,
    title: str = "Parent vs Child Validation Score",
):
    """Create scatter plot of parent vs child validation scores with child_idx labels."""
    if not pairs:
        print("No data to plot")
        return

    parent_scores = [p["parent_score"] for p in pairs]
    child_scores = [p["child_score"] for p in pairs]
    child_indices = [p["child_idx"] for p in pairs]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Color by improvement
    colors = ["green" if c > p else "red" for p, c in zip(parent_scores, child_scores)]
    ax.scatter(parent_scores, child_scores, c=colors, alpha=0.6, s=80)

    # Add child_idx labels with small random offset to avoid overlap
    np.random.seed(42)
    for i, (px, py, idx) in enumerate(zip(parent_scores, child_scores, child_indices)):
        # Add small random offset to prevent overlap
        offset_x = np.random.uniform(-0.015, 0.015)
        offset_y = np.random.uniform(-0.015, 0.015)
        ax.annotate(
            str(idx),
            (px + offset_x, py + offset_y),
            fontsize=6,
            alpha=0.8,
            ha="center",
            va="bottom",
        )

    # Add diagonal line
    max_val = max(max(parent_scores), max(child_scores))
    min_val = min(min(parent_scores), min(child_scores))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="No change (y=x)")

    ax.set_xlabel("Parent Validation Score", fontsize=12)
    ax.set_ylabel("Child Validation Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    input_path = "/Users/panwenbo/Repos/gepa/artifacts/mutation_analysis.json"
    output_path = "/Users/panwenbo/Repos/gepa/artifacts/valset_scatter_with_labels.png"

    print(f"Loading data from {input_path}")
    pairs = load_mutation_data(input_path)
    print(f"Loaded {len(pairs)} parent-child pairs")

    plot_valset_scatter_with_labels(
        pairs,
        output_path,
        "Parent vs Child Validation Score (with child_idx labels)",
    )


if __name__ == "__main__":
    main()
