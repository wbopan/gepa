"""Analyze parent features that predict strong child performance."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy import stats


def fetch_all_candidates(entity: str, project: str) -> dict[str, list[dict]]:
    """Fetch all candidate data including prompts from all runs."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    all_candidates = {}

    for run in runs:
        run_name = run.display_name
        print(f"Fetching {run_name}...")

        # Get the latest version of candidate_prompts table
        latest_candidates = []
        for artifact in run.logged_artifacts():
            if "candidate_prompts" in artifact.name:
                artifact_dir = artifact.download()
                table_file = Path(artifact_dir) / "candidate_prompts.table.json"
                if table_file.exists():
                    with open(table_file) as f:
                        table_data = json.load(f)
                    columns = table_data["columns"]
                    rows = [dict(zip(columns, row)) for row in table_data["data"]]
                    if len(rows) > len(latest_candidates):
                        latest_candidates = rows

        if latest_candidates:
            all_candidates[run_name] = latest_candidates
            print(f"  {len(latest_candidates)} candidates")

    return all_candidates


def analyze_parent_child_features(candidates: list[dict]) -> list[dict]:
    """Build feature analysis for each parent-child pair."""
    # Build lookup: candidate_idx -> candidate info
    by_idx = {c["candidate_idx"]: c for c in candidates}

    pairs = []
    for child in candidates:
        parent_idx = child.get("parent_idx")
        if parent_idx is None or parent_idx not in by_idx:
            continue

        parent = by_idx[parent_idx]

        # Parse prompt content
        try:
            parent_content = json.loads(parent["candidate_content"])
            child_content = json.loads(child["candidate_content"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Calculate features
        parent_prompt = parent_content.get("system_prompt", "")
        child_prompt = child_content.get("system_prompt", "")

        pairs.append(
            {
                "parent_idx": parent_idx,
                "child_idx": child["candidate_idx"],
                "parent_score": parent["valset_aggregate_score"],
                "child_score": child["valset_aggregate_score"],
                "parent_length": len(parent_prompt),
                "child_length": len(child_prompt),
                "length_change": len(child_prompt) - len(parent_prompt),
                "parent_prompt": parent_prompt,
                "child_prompt": child_prompt,
            }
        )

    return pairs


def compute_correlations(pairs: list[dict]) -> dict:
    """Compute correlations between parent features and child score."""
    parent_scores = np.array([p["parent_score"] for p in pairs])
    child_scores = np.array([p["child_score"] for p in pairs])
    parent_lengths = np.array([p["parent_length"] for p in pairs])
    improvements = child_scores - parent_scores

    results = {}

    # Correlation: parent_score vs child_score
    r, p = stats.pearsonr(parent_scores, child_scores)
    results["parent_score_vs_child_score"] = {"r": r, "p": p}

    # Correlation: parent_length vs child_score
    r, p = stats.pearsonr(parent_lengths, child_scores)
    results["parent_length_vs_child_score"] = {"r": r, "p": p}

    # Correlation: parent_score vs improvement
    r, p = stats.pearsonr(parent_scores, improvements)
    results["parent_score_vs_improvement"] = {"r": r, "p": p}

    # Correlation: parent_length vs improvement
    r, p = stats.pearsonr(parent_lengths, improvements)
    results["parent_length_vs_improvement"] = {"r": r, "p": p}

    return results


def group_by_parent(pairs: list[dict]) -> dict:
    """Group pairs by parent and compute statistics."""
    by_parent = defaultdict(list)
    for p in pairs:
        by_parent[p["parent_idx"]].append(p)

    parent_stats = []
    for parent_idx, children in by_parent.items():
        if not children:
            continue

        child_scores = [c["child_score"] for c in children]
        parent_stats.append(
            {
                "parent_idx": parent_idx,
                "parent_score": children[0]["parent_score"],
                "parent_length": children[0]["parent_length"],
                "parent_prompt": children[0]["parent_prompt"],
                "num_children": len(children),
                "mean_child_score": np.mean(child_scores),
                "max_child_score": np.max(child_scores),
                "min_child_score": np.min(child_scores),
                "std_child_score": np.std(child_scores) if len(child_scores) > 1 else 0,
                "improvement_rate": sum(1 for c in children if c["child_score"] > c["parent_score"]) / len(children),
            }
        )

    return parent_stats


def plot_correlations(pairs: list[dict], output_dir: Path):
    """Create correlation plots."""
    parent_scores = [p["parent_score"] for p in pairs]
    child_scores = [p["child_score"] for p in pairs]
    parent_lengths = [p["parent_length"] for p in pairs]
    improvements = [p["child_score"] - p["parent_score"] for p in pairs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Parent score vs Child score
    ax = axes[0, 0]
    ax.scatter(parent_scores, child_scores, alpha=0.5, s=30)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    r = np.corrcoef(parent_scores, child_scores)[0, 1]
    ax.set_xlabel("Parent Score")
    ax.set_ylabel("Child Score")
    ax.set_title(f"Parent Score vs Child Score (r={r:.3f})")
    ax.grid(True, alpha=0.3)

    # 2. Parent length vs Child score
    ax = axes[0, 1]
    ax.scatter(parent_lengths, child_scores, alpha=0.5, s=30)
    r = np.corrcoef(parent_lengths, child_scores)[0, 1]
    ax.set_xlabel("Parent Prompt Length (chars)")
    ax.set_ylabel("Child Score")
    ax.set_title(f"Parent Length vs Child Score (r={r:.3f})")
    ax.grid(True, alpha=0.3)

    # 3. Parent score vs Improvement
    ax = axes[1, 0]
    colors = ["green" if i > 0 else "red" for i in improvements]
    ax.scatter(parent_scores, improvements, c=colors, alpha=0.5, s=30)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    r = np.corrcoef(parent_scores, improvements)[0, 1]
    ax.set_xlabel("Parent Score")
    ax.set_ylabel("Improvement (Child - Parent)")
    ax.set_title(f"Parent Score vs Improvement (r={r:.3f})")
    ax.grid(True, alpha=0.3)

    # 4. Parent length vs Improvement
    ax = axes[1, 1]
    ax.scatter(parent_lengths, improvements, c=colors, alpha=0.5, s=30)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    r = np.corrcoef(parent_lengths, improvements)[0, 1]
    ax.set_xlabel("Parent Prompt Length (chars)")
    ax.set_ylabel("Improvement (Child - Parent)")
    ax.set_title(f"Parent Length vs Improvement (r={r:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "parent_features_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation plots")


def plot_by_bucket(pairs: list[dict], output_dir: Path):
    """Plot statistics by parent score and length buckets."""
    # By parent score bucket
    score_buckets = defaultdict(list)
    for p in pairs:
        bucket = round(p["parent_score"] * 10) / 10  # Round to 0.1
        score_buckets[bucket].append(p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score bucket analysis
    ax = axes[0]
    buckets = sorted(score_buckets.keys())
    means = [np.mean([p["child_score"] for p in score_buckets[b]]) for b in buckets]
    maxes = [np.max([p["child_score"] for p in score_buckets[b]]) for b in buckets]
    counts = [len(score_buckets[b]) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35
    ax.bar(x - width / 2, means, width, label="Mean Child Score", color="steelblue")
    ax.bar(x + width / 2, maxes, width, label="Max Child Score", color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}\n(n={c})" for b, c in zip(buckets, counts)])
    ax.set_xlabel("Parent Score Bucket")
    ax.set_ylabel("Child Score")
    ax.set_title("Child Score by Parent Score Bucket")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Length bucket analysis
    ax = axes[1]
    length_buckets = defaultdict(list)
    for p in pairs:
        # Bucket by 500 chars
        bucket = (p["parent_length"] // 500) * 500
        length_buckets[bucket].append(p)

    buckets = sorted(length_buckets.keys())
    means = [np.mean([p["child_score"] for p in length_buckets[b]]) for b in buckets]
    maxes = [np.max([p["child_score"] for p in length_buckets[b]]) for b in buckets]
    counts = [len(length_buckets[b]) for b in buckets]

    x = np.arange(len(buckets))
    ax.bar(x - width / 2, means, width, label="Mean Child Score", color="steelblue")
    ax.bar(x + width / 2, maxes, width, label="Max Child Score", color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}-{b+500}\n(n={c})" for b, c in zip(buckets, counts)], rotation=45, ha="right")
    ax.set_xlabel("Parent Length Bucket (chars)")
    ax.set_ylabel("Child Score")
    ax.set_title("Child Score by Parent Length Bucket")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "parent_buckets_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bucket analysis plots")


def find_best_parents(parent_stats: list[dict], top_n: int = 5) -> list[dict]:
    """Find parents that produced the best children."""
    # Sort by max child score
    sorted_by_max = sorted(parent_stats, key=lambda x: x["max_child_score"], reverse=True)
    return sorted_by_max[:top_n]


def find_most_improving_parents(parent_stats: list[dict], top_n: int = 5) -> list[dict]:
    """Find parents with highest improvement rate."""
    # Filter for parents with at least 2 children
    filtered = [p for p in parent_stats if p["num_children"] >= 2]
    sorted_by_improvement = sorted(filtered, key=lambda x: x["improvement_rate"], reverse=True)
    return sorted_by_improvement[:top_n]


def main():
    entity = "bmpixel"
    project = "gepa-boost"

    output_dir = Path("/Users/panwenbo/Repos/gepa/artifacts")
    output_dir.mkdir(exist_ok=True)

    # Fetch data
    all_candidates = fetch_all_candidates(entity, project)

    # Combine all pairs
    all_pairs = []
    for run_name, candidates in all_candidates.items():
        pairs = analyze_parent_child_features(candidates)
        all_pairs.extend(pairs)

    print(f"\nTotal parent-child pairs: {len(all_pairs)}")

    # Compute correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    correlations = compute_correlations(all_pairs)
    for name, result in correlations.items():
        sig = "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
        print(f"{name}: r={result['r']:.4f}, p={result['p']:.4e} {sig}")

    # Group by parent
    parent_stats = group_by_parent(all_pairs)

    # Compute correlations at parent level
    print("\n" + "=" * 70)
    print("PARENT-LEVEL CORRELATIONS (grouped by parent)")
    print("=" * 70)

    parent_scores = np.array([p["parent_score"] for p in parent_stats])
    parent_lengths = np.array([p["parent_length"] for p in parent_stats])
    max_child_scores = np.array([p["max_child_score"] for p in parent_stats])
    mean_child_scores = np.array([p["mean_child_score"] for p in parent_stats])
    improvement_rates = np.array([p["improvement_rate"] for p in parent_stats])

    print("\nCorrelations with MAX child score:")
    r, p = stats.pearsonr(parent_scores, max_child_scores)
    print(f"  parent_score vs max_child_score: r={r:.4f}, p={p:.4e}")
    r, p = stats.pearsonr(parent_lengths, max_child_scores)
    print(f"  parent_length vs max_child_score: r={r:.4f}, p={p:.4e}")

    print("\nCorrelations with MEAN child score:")
    r, p = stats.pearsonr(parent_scores, mean_child_scores)
    print(f"  parent_score vs mean_child_score: r={r:.4f}, p={p:.4e}")
    r, p = stats.pearsonr(parent_lengths, mean_child_scores)
    print(f"  parent_length vs mean_child_score: r={r:.4f}, p={p:.4e}")

    print("\nCorrelations with IMPROVEMENT RATE:")
    r, p = stats.pearsonr(parent_scores, improvement_rates)
    print(f"  parent_score vs improvement_rate: r={r:.4f}, p={p:.4e}")
    r, p = stats.pearsonr(parent_lengths, improvement_rates)
    print(f"  parent_length vs improvement_rate: r={r:.4f}, p={p:.4e}")

    # Plot
    plot_correlations(all_pairs, output_dir)
    plot_by_bucket(all_pairs, output_dir)

    # Find best parents
    print("\n" + "=" * 70)
    print("TOP 5 PARENTS BY MAX CHILD SCORE")
    print("=" * 70)

    best_parents = find_best_parents(parent_stats)
    for i, p in enumerate(best_parents, 1):
        print(f"\n{i}. Parent #{p['parent_idx']}")
        print(f"   Parent score: {p['parent_score']:.3f}")
        print(f"   Parent length: {p['parent_length']} chars")
        print(f"   Max child score: {p['max_child_score']:.3f}")
        print(f"   Mean child score: {p['mean_child_score']:.3f}")
        print(f"   Num children: {p['num_children']}")
        print(f"   Improvement rate: {p['improvement_rate']*100:.1f}%")
        print(f"   Prompt preview: {p['parent_prompt'][:200]}...")

    # Find most improving parents
    print("\n" + "=" * 70)
    print("TOP 5 PARENTS BY IMPROVEMENT RATE (min 2 children)")
    print("=" * 70)

    improving_parents = find_most_improving_parents(parent_stats)
    for i, p in enumerate(improving_parents, 1):
        print(f"\n{i}. Parent #{p['parent_idx']}")
        print(f"   Parent score: {p['parent_score']:.3f}")
        print(f"   Parent length: {p['parent_length']} chars")
        print(f"   Max child score: {p['max_child_score']:.3f}")
        print(f"   Mean child score: {p['mean_child_score']:.3f}")
        print(f"   Num children: {p['num_children']}")
        print(f"   Improvement rate: {p['improvement_rate']*100:.1f}%")
        print(f"   Prompt preview: {p['parent_prompt'][:200]}...")

    # Save detailed analysis
    analysis_path = output_dir / "parent_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(
            {
                "correlations": {k: {"r": v["r"], "p": float(v["p"])} for k, v in correlations.items()},
                "parent_stats": [
                    {k: v for k, v in p.items() if k != "parent_prompt"}  # Exclude full prompt
                    for p in parent_stats
                ],
                "best_parents": [p["parent_idx"] for p in best_parents],
            },
            f,
            indent=2,
        )
    print(f"\nSaved analysis to {analysis_path}")

    # Print bucket statistics
    print("\n" + "=" * 70)
    print("STATISTICS BY PARENT SCORE BUCKET")
    print("=" * 70)

    score_buckets = defaultdict(list)
    for p in all_pairs:
        bucket = round(p["parent_score"] * 10) / 10
        score_buckets[bucket].append(p)

    print(f"\n{'Bucket':<8} {'N':<6} {'Mean':<8} {'Max':<8} {'Improve%':<10}")
    print("-" * 45)
    for bucket in sorted(score_buckets.keys()):
        pairs_in_bucket = score_buckets[bucket]
        n = len(pairs_in_bucket)
        mean_child = np.mean([p["child_score"] for p in pairs_in_bucket])
        max_child = np.max([p["child_score"] for p in pairs_in_bucket])
        improve_rate = sum(1 for p in pairs_in_bucket if p["child_score"] > p["parent_score"]) / n * 100
        print(f"{bucket:<8.1f} {n:<6} {mean_child:<8.3f} {max_child:<8.3f} {improve_rate:<10.1f}%")

    print("\n" + "=" * 70)
    print("STATISTICS BY PARENT LENGTH BUCKET")
    print("=" * 70)

    length_buckets = defaultdict(list)
    for p in all_pairs:
        bucket = (p["parent_length"] // 500) * 500
        length_buckets[bucket].append(p)

    print(f"\n{'Length':<12} {'N':<6} {'Mean':<8} {'Max':<8} {'Improve%':<10}")
    print("-" * 50)
    for bucket in sorted(length_buckets.keys()):
        pairs_in_bucket = length_buckets[bucket]
        n = len(pairs_in_bucket)
        mean_child = np.mean([p["child_score"] for p in pairs_in_bucket])
        max_child = np.max([p["child_score"] for p in pairs_in_bucket])
        improve_rate = sum(1 for p in pairs_in_bucket if p["child_score"] > p["parent_score"]) / n * 100
        print(f"{bucket}-{bucket+500:<6} {n:<6} {mean_child:<8.3f} {max_child:<8.3f} {improve_rate:<10.1f}%")


if __name__ == "__main__":
    main()
