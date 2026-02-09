"""Fetch validation set scores from wandb and plot parent-child relationship."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def fetch_candidate_prompts_table(entity: str, project: str, run_name: str) -> list[dict]:
    """Fetch candidate_prompts table from a wandb run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_name}")

    # Get the candidate_prompts artifact
    artifacts = run.logged_artifacts()
    candidate_data = []

    for artifact in artifacts:
        if "candidate_prompts" in artifact.name:
            # Download and read the table
            table_path = artifact.download()
            table_file = Path(table_path) / "candidate_prompts.table.json"
            if table_file.exists():
                with open(table_file) as f:
                    table_data = json.load(f)
                    columns = table_data["columns"]
                    for row in table_data["data"]:
                        candidate_data.append(dict(zip(columns, row)))

    return candidate_data


def fetch_all_runs_data(entity: str, project: str) -> dict[str, list[dict]]:
    """Fetch candidate data from all runs in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    all_data = {}
    for run in runs:
        print(f"Fetching data from run: {run.name} ({run.display_name})")
        try:
            # Try to get history with candidate/selected_idx and val/new_score
            history = run.history(keys=["gepa_iteration", "candidate/selected_idx", "val/new_score", "val/best_agg"])
            if not history.empty:
                all_data[run.display_name] = history.to_dict("records")
        except Exception as e:
            print(f"  Error: {e}")

    return all_data


def build_parent_child_valscores(run_data: list[dict]) -> list[dict]:
    """Build parent-child validation score pairs from run history.

    The key insight: when a new candidate is discovered,
    - val/new_score is the child's validation score
    - We need to look up the parent's validation score from prior accepted candidates

    Since candidate_prompts table has (candidate_idx, valset_aggregate_score, parent_idx),
    we can reconstruct the relationship.
    """
    # candidate_idx -> valset_aggregate_score mapping
    scores = {}
    pairs = []

    for row in run_data:
        candidate_idx = row.get("candidate_idx")
        valset_score = row.get("valset_aggregate_score")
        parent_idx = row.get("parent_idx")

        if candidate_idx is not None and valset_score is not None:
            scores[candidate_idx] = valset_score

            if parent_idx is not None and parent_idx in scores:
                pairs.append(
                    {
                        "parent_idx": parent_idx,
                        "child_idx": candidate_idx,
                        "parent_score": scores[parent_idx],
                        "child_score": valset_score,
                    }
                )

    return pairs


def plot_valset_scatter(pairs: list[dict], output_path: str, title: str = "Parent vs Child Validation Score"):
    """Create scatter plot of parent vs child validation scores."""
    if not pairs:
        print("No data to plot")
        return

    parent_scores = [p["parent_score"] for p in pairs]
    child_scores = [p["child_score"] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by improvement
    colors = ["green" if c > p else "red" for p, c in zip(parent_scores, child_scores)]
    ax.scatter(parent_scores, child_scores, c=colors, alpha=0.6, s=80)

    # Add diagonal line
    max_val = max(max(parent_scores), max(child_scores))
    min_val = min(min(parent_scores), min(child_scores))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="No change (y=x)")

    # Add jitter to see overlapping points
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
    entity = "bmpixel"
    project = "gepa-boost"

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    output_dir = Path("/Users/panwenbo/Repos/gepa/artifacts")
    output_dir.mkdir(exist_ok=True)

    all_pairs = []

    for run in runs:
        print(f"\n{'='*60}")
        print(f"Run: {run.name} ({run.display_name})")
        print(f"{'='*60}")

        # Try to fetch the candidate_prompts table directly from history
        try:
            # Get all logged tables
            for artifact in run.logged_artifacts():
                if "run-" in artifact.name and "candidate_prompts" in str(artifact.name):
                    print(f"  Found artifact: {artifact.name}")
                    artifact_dir = artifact.download()
                    table_file = Path(artifact_dir) / "candidate_prompts.table.json"
                    if table_file.exists():
                        with open(table_file) as f:
                            table_data = json.load(f)
                        columns = table_data["columns"]
                        rows = [dict(zip(columns, row)) for row in table_data["data"]]
                        print(f"  Loaded {len(rows)} candidates")

                        # Build parent-child pairs
                        pairs = build_parent_child_valscores(rows)
                        print(f"  Found {len(pairs)} parent-child pairs")
                        all_pairs.extend(pairs)

                        # Plot individual run
                        if pairs:
                            plot_valset_scatter(
                                pairs,
                                str(output_dir / f"valset_scatter_{run.display_name}.png"),
                                f"Parent vs Child Validation Score ({run.display_name})",
                            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    # Plot combined
    if all_pairs:
        print(f"\n{'='*60}")
        print(f"Combined: {len(all_pairs)} parent-child pairs")
        print(f"{'='*60}")
        plot_valset_scatter(all_pairs, str(output_dir / "valset_scatter_combined.png"), "Parent vs Child Validation Score (All Runs)")

        # Compute statistics
        parent_scores = [p["parent_score"] for p in all_pairs]
        child_scores = [p["child_score"] for p in all_pairs]
        improvements = [c - p for p, c in zip(parent_scores, child_scores)]

        print(f"\nStatistics:")
        print(f"  Total pairs: {len(all_pairs)}")
        print(f"  Mean parent score: {np.mean(parent_scores):.4f}")
        print(f"  Mean child score: {np.mean(child_scores):.4f}")
        print(f"  Mean improvement: {np.mean(improvements):.4f}")
        print(f"  Correlation: {np.corrcoef(parent_scores, child_scores)[0, 1]:.4f}")
        print(f"  Improvement rate: {sum(1 for i in improvements if i > 0) / len(improvements) * 100:.1f}%")

        # Group by parent score buckets
        print("\nBy parent score bucket:")
        buckets = {}
        for p in all_pairs:
            bucket = round(p["parent_score"], 1)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(p["child_score"])

        for bucket in sorted(buckets.keys()):
            children = buckets[bucket]
            improved = sum(1 for c in children if c > bucket)
            print(f"  Parent {bucket:.1f}: n={len(children)}, mean_child={np.mean(children):.3f}, improved={improved}/{len(children)} ({improved/len(children)*100:.0f}%)")


if __name__ == "__main__":
    main()
