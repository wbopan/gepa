"""Plot parent-child score relationship from GEPA mutation traces."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


def load_data(json_path: str) -> list[dict]:
    """Load extracted scores from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def plot_scatter(data: list[dict], output_path: str):
    """Create scatter plot of parent vs child scores."""
    # Filter valid data points
    valid_data = [d for d in data if d["parent_score"] is not None and d["child_score"] is not None]

    # Separate by accepted status
    accepted = [(d["parent_score"], d["child_score"]) for d in valid_data if d["accepted"]]
    rejected = [(d["parent_score"], d["child_score"]) for d in valid_data if not d["accepted"]]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot rejected first (so accepted are on top)
    if rejected:
        rej_x, rej_y = zip(*rejected)
        ax.scatter(rej_x, rej_y, c="red", alpha=0.4, label=f"Rejected (n={len(rejected)})", s=50)

    if accepted:
        acc_x, acc_y = zip(*accepted)
        ax.scatter(acc_x, acc_y, c="green", alpha=0.6, label=f"Accepted (n={len(accepted)})", s=50)

    # Add diagonal line (y = x, no change)
    max_val = max(max(d["parent_score"] for d in valid_data), max(d["child_score"] for d in valid_data))
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="No change (y=x)")

    ax.set_xlabel("Parent Score (subsample sum)", fontsize=12)
    ax.set_ylabel("Child Score (subsample sum)", fontsize=12)
    ax.set_title("Parent vs Child Score Relationship in GEPA Mutations", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Make it square
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def plot_heatmap(data: list[dict], output_path: str):
    """Create heatmap of parent-child score transitions."""
    valid_data = [d for d in data if d["parent_score"] is not None and d["child_score"] is not None]

    # Count transitions
    transitions = Counter((int(d["parent_score"]), int(d["child_score"])) for d in valid_data)

    # Find range
    all_scores = [int(d["parent_score"]) for d in valid_data] + [int(d["child_score"]) for d in valid_data]
    min_score, max_score = min(all_scores), max(all_scores)
    score_range = range(min_score, max_score + 1)

    # Create matrix
    matrix = np.zeros((len(score_range), len(score_range)))
    for (parent, child), count in transitions.items():
        matrix[child - min_score, parent - min_score] = count

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", origin="lower")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=12)

    # Set ticks
    ax.set_xticks(range(len(score_range)))
    ax.set_yticks(range(len(score_range)))
    ax.set_xticklabels(score_range)
    ax.set_yticklabels(score_range)

    # Add text annotations
    for i in range(len(score_range)):
        for j in range(len(score_range)):
            if matrix[i, j] > 0:
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=text_color, fontsize=9)

    # Draw diagonal
    ax.plot([-0.5, len(score_range) - 0.5], [-0.5, len(score_range) - 0.5], "k--", alpha=0.5, linewidth=2)

    ax.set_xlabel("Parent Score", fontsize=12)
    ax.set_ylabel("Child Score", fontsize=12)
    ax.set_title("Transition Heatmap: Parent → Child Score", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


def compute_statistics(data: list[dict]) -> dict:
    """Compute summary statistics."""
    valid_data = [d for d in data if d["parent_score"] is not None and d["child_score"] is not None]

    # Group by parent score
    by_parent = {}
    for d in valid_data:
        parent = d["parent_score"]
        if parent not in by_parent:
            by_parent[parent] = {"children": [], "accepted": 0, "total": 0}
        by_parent[parent]["children"].append(d["child_score"])
        by_parent[parent]["total"] += 1
        if d["accepted"]:
            by_parent[parent]["accepted"] += 1

    print("\n" + "=" * 70)
    print("Parent-Child Score Statistics")
    print("=" * 70)

    for parent in sorted(by_parent.keys()):
        info = by_parent[parent]
        children = info["children"]
        mean_child = np.mean(children)
        improved_count = sum(1 for c in children if c > parent)
        improved_rate = improved_count / len(children) * 100

        print(f"\nParent Score: {parent:.1f}")
        print(f"  Total mutations: {info['total']}")
        print(f"  Mean child score: {mean_child:.3f}")
        print(f"  Improvement rate: {improved_rate:.1f}% ({improved_count}/{len(children)})")
        print(f"  Acceptance rate: {info['accepted']/info['total']*100:.1f}% ({info['accepted']}/{info['total']})")

    # Overall statistics
    parent_scores = [d["parent_score"] for d in valid_data]
    child_scores = [d["child_score"] for d in valid_data]
    improvements = [c - p for p, c in zip(parent_scores, child_scores)]

    print("\n" + "-" * 70)
    print("Overall Statistics")
    print("-" * 70)
    print(f"Total mutations: {len(valid_data)}")
    print(f"Mean parent score: {np.mean(parent_scores):.3f}")
    print(f"Mean child score: {np.mean(child_scores):.3f}")
    print(f"Mean improvement: {np.mean(improvements):.3f}")
    print(f"Correlation (parent, child): {np.corrcoef(parent_scores, child_scores)[0, 1]:.3f}")

    # Improvement rate by parent score bucket
    print(f"\nImprovement rate: {sum(1 for i in improvements if i > 0) / len(improvements) * 100:.1f}%")
    print(f"No change rate: {sum(1 for i in improvements if i == 0) / len(improvements) * 100:.1f}%")
    print(f"Degradation rate: {sum(1 for i in improvements if i < 0) / len(improvements) * 100:.1f}%")


def main():
    # Load data
    json_path = "/tmp/parent_child_scores.json"
    data = load_data(json_path)
    print(f"Loaded {len(data)} records")

    # Output directory
    output_dir = Path("/Users/panwenbo/Repos/gepa/artifacts")
    output_dir.mkdir(exist_ok=True)

    # Create plots
    plot_scatter(data, str(output_dir / "parent_child_scatter.png"))
    plot_heatmap(data, str(output_dir / "parent_child_heatmap.png"))

    # Compute statistics
    compute_statistics(data)


if __name__ == "__main__":
    main()
