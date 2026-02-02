"""Statistical analysis of existing GEPA runs to determine if observed Pareto frontier
differences are due to method or random variation.

Analyzes validation set scores from adaboost, bayesian, and baseline methods.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_scores(data_dir: Path) -> dict[str, np.ndarray]:
    """Load validation scores for each method.

    Returns dict mapping method name to (n_candidates, n_val_samples) array.
    """
    scores = {}
    for method in ["adaboost", "bayesian", "baseline"]:
        filepath = data_dir / f"valset_scores_{method}.json"
        with open(filepath) as f:
            data = json.load(f)

        # Convert to numpy array: rows=candidates, cols=val_samples
        n_candidates = len(data)
        n_samples = len(next(iter(data.values())))
        arr = np.zeros((n_candidates, n_samples))

        for cand_idx, (_cand_id, sample_scores) in enumerate(data.items()):
            for sample_id, score in sample_scores.items():
                arr[cand_idx, int(sample_id)] = score

        scores[method] = arr
        print(f"Loaded {method}: {arr.shape[0]} candidates x {arr.shape[1]} val samples")

    return scores


def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if candidate a is dominated by candidate b.

    b dominates a if b >= a on all samples AND b > a on at least one sample.
    """
    return bool(np.all(b >= a) and np.any(b > a))


def compute_pareto_frontier(
    scores: np.ndarray, sample_indices: np.ndarray | None = None
) -> tuple[list[int], float]:
    """Compute Pareto frontier for given scores.

    Args:
        scores: (n_candidates, n_samples) array
        sample_indices: Optional indices of samples to use (for bootstrap)

    Returns:
        List of frontier candidate indices and frontier ratio
    """
    if sample_indices is not None:
        scores = scores[:, sample_indices]

    n_candidates = scores.shape[0]
    is_frontier = [True] * n_candidates

    for i in range(n_candidates):
        if not is_frontier[i]:
            continue
        for j in range(n_candidates):
            if i == j or not is_frontier[j]:
                continue
            if is_dominated(scores[i], scores[j]):
                is_frontier[i] = False
                break

    frontier_indices = [i for i, on_frontier in enumerate(is_frontier) if on_frontier]
    frontier_ratio = len(frontier_indices) / n_candidates
    return frontier_indices, frontier_ratio


def bootstrap_frontier_ratio(
    scores: np.ndarray, n_bootstrap: int = 1000, seed: int = 42
) -> tuple[float, float, float, list[float]]:
    """Compute bootstrap distribution of frontier ratio.

    Returns:
        mean, lower 95% CI, upper 95% CI, full distribution
    """
    rng = np.random.default_rng(seed)
    n_samples = scores.shape[1]
    bootstrap_ratios = []

    for _ in range(n_bootstrap):
        # Resample validation samples with replacement
        resampled_indices = rng.choice(n_samples, size=n_samples, replace=True)
        _, ratio = compute_pareto_frontier(scores, resampled_indices)
        bootstrap_ratios.append(ratio)

    bootstrap_ratios = np.array(bootstrap_ratios)
    mean = float(np.mean(bootstrap_ratios))
    ci_lower = float(np.percentile(bootstrap_ratios, 2.5))
    ci_upper = float(np.percentile(bootstrap_ratios, 97.5))

    return mean, ci_lower, ci_upper, bootstrap_ratios.tolist()


def permutation_test_frontier(
    scores_a: np.ndarray, scores_b: np.ndarray, n_perm: int = 10000, seed: int = 42
) -> tuple[float, float, list[float]]:
    """Permutation test for difference in frontier ratios.

    H0: Method labels don't affect frontier ratio.

    Returns:
        observed difference, p-value, null distribution
    """
    rng = np.random.default_rng(seed)

    # Observed difference
    _, ratio_a = compute_pareto_frontier(scores_a)
    _, ratio_b = compute_pareto_frontier(scores_b)
    observed_diff = ratio_a - ratio_b

    # Combine candidates
    combined = np.vstack([scores_a, scores_b])
    n_a = scores_a.shape[0]
    n_total = combined.shape[0]

    # Permutation null distribution
    null_diffs = []
    for _ in range(n_perm):
        perm_indices = rng.permutation(n_total)
        perm_a = combined[perm_indices[:n_a]]
        perm_b = combined[perm_indices[n_a:]]
        _, perm_ratio_a = compute_pareto_frontier(perm_a)
        _, perm_ratio_b = compute_pareto_frontier(perm_b)
        null_diffs.append(perm_ratio_a - perm_ratio_b)

    # Two-tailed p-value
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value, null_diffs.tolist()


def per_sample_comparison(scores_dict: dict[str, np.ndarray]) -> dict:
    """Compare methods on each validation sample.

    Returns per-sample success rates and pairwise comparisons.
    """
    methods = list(scores_dict.keys())
    n_samples = scores_dict[methods[0]].shape[1]

    # Per-sample success rates
    sample_rates = {method: scores_dict[method].mean(axis=0) for method in methods}

    # Count wins per sample (which method has highest success rate)
    wins = defaultdict(int)
    for sample_idx in range(n_samples):
        rates = {m: sample_rates[m][sample_idx] for m in methods}
        max_rate = max(rates.values())
        winners = [m for m, r in rates.items() if r == max_rate]
        for w in winners:
            wins[w] += 1.0 / len(winners)

    # Paired comparison: For each pair, count samples where one beats the other
    pairwise = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            m1_wins = np.sum(sample_rates[m1] > sample_rates[m2])
            m2_wins = np.sum(sample_rates[m2] > sample_rates[m1])
            ties = n_samples - m1_wins - m2_wins
            pairwise[f"{m1}_vs_{m2}"] = {"m1_wins": int(m1_wins), "m2_wins": int(m2_wins), "ties": int(ties)}

    return {
        "sample_rates": {m: sample_rates[m].tolist() for m in methods},
        "sample_wins": dict(wins),
        "pairwise_comparisons": pairwise,
    }


def candidate_quality_test(scores_dict: dict[str, np.ndarray]) -> dict:
    """Compare candidate quality distributions across methods.

    Uses mean score per candidate as quality metric.
    """
    methods = list(scores_dict.keys())
    quality = {method: scores_dict[method].mean(axis=1) for method in methods}

    results = {
        "quality_stats": {
            method: {
                "mean": float(np.mean(quality[method])),
                "std": float(np.std(quality[method])),
                "median": float(np.median(quality[method])),
                "min": float(np.min(quality[method])),
                "max": float(np.max(quality[method])),
                "n_candidates": len(quality[method]),
            }
            for method in methods
        },
        "pairwise_tests": {},
    }

    # Mann-Whitney U tests
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            stat, p_value = stats.mannwhitneyu(quality[m1], quality[m2], alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(quality[m1]), len(quality[m2])
            effect_size = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial correlation
            results["pairwise_tests"][f"{m1}_vs_{m2}"] = {
                "mann_whitney_u": float(stat),
                "p_value": float(p_value),
                "effect_size_rank_biserial": float(effect_size),
            }

    # KS tests
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            stat, p_value = stats.ks_2samp(quality[m1], quality[m2])
            results["pairwise_tests"][f"{m1}_vs_{m2}"]["ks_statistic"] = float(stat)
            results["pairwise_tests"][f"{m1}_vs_{m2}"]["ks_p_value"] = float(p_value)

    return results


def domination_analysis(scores_dict: dict[str, np.ndarray]) -> dict:
    """Analyze domination structure within each method's candidates.

    Computes domination graph statistics.
    """
    results = {}

    for method, scores in scores_dict.items():
        n_candidates = scores.shape[0]

        # Count how many candidates each candidate dominates
        dominates_count = np.zeros(n_candidates)
        dominated_by_count = np.zeros(n_candidates)

        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j and is_dominated(scores[j], scores[i]):
                    dominates_count[i] += 1
                    dominated_by_count[j] += 1

        # Gini coefficient of domination counts
        dom_sorted = np.sort(dominates_count)
        n = len(dom_sorted)
        cumsum = np.cumsum(dom_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * dom_sorted))) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0

        results[method] = {
            "n_candidates": n_candidates,
            "dominates_count": {
                "mean": float(np.mean(dominates_count)),
                "max": int(np.max(dominates_count)),
                "std": float(np.std(dominates_count)),
            },
            "dominated_by_count": {
                "mean": float(np.mean(dominated_by_count)),
                "max": int(np.max(dominated_by_count)),
                "std": float(np.std(dominated_by_count)),
            },
            "gini_coefficient": float(gini),
            "max_dominator_ratio": float(np.max(dominates_count) / (n_candidates - 1)) if n_candidates > 1 else 0,
        }

    return results


def create_visualizations(
    scores_dict: dict[str, np.ndarray],
    bootstrap_results: dict,
    _quality_results: dict,
    output_dir: Path,
) -> None:
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualizations")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    methods = list(scores_dict.keys())
    colors = {"adaboost": "#1f77b4", "bayesian": "#ff7f0e", "baseline": "#2ca02c"}

    # 1. Bootstrap distribution of frontier ratios
    _fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        data = bootstrap_results[method]["distribution"]
        ax.hist(data, bins=30, alpha=0.5, label=method, color=colors[method])
        ax.axvline(bootstrap_results[method]["mean"], color=colors[method], linestyle="--", linewidth=2)

    ax.set_xlabel("Pareto Frontier Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution of Pareto Frontier Ratios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "bootstrap_frontier_distribution.png", dpi=150)
    plt.close()

    # 2. Candidate quality distribution (boxplot)
    _fig, ax = plt.subplots(figsize=(8, 6))
    quality_data = [scores_dict[m].mean(axis=1) for m in methods]
    bp = ax.boxplot(quality_data, tick_labels=methods, patch_artist=True)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)

    ax.set_ylabel("Mean Candidate Score (across val samples)")
    ax.set_title("Candidate Quality Distribution by Method")
    plt.tight_layout()
    plt.savefig(figures_dir / "candidate_quality_boxplot.png", dpi=150)
    plt.close()

    # 3. Per-sample success rate heatmap
    n_samples = scores_dict[methods[0]].shape[1]
    sample_rates = np.array([scores_dict[m].mean(axis=0) for m in methods])

    _fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(sample_rates, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(range(n_samples))
    ax.set_xlabel("Validation Sample ID")
    ax.set_ylabel("Method")
    ax.set_title("Per-Sample Success Rate by Method")
    plt.colorbar(im, ax=ax, label="Success Rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_sample_heatmap.png", dpi=150)
    plt.close()

    # 4. CI comparison plot
    _fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(methods))
    means = [bootstrap_results[m]["mean"] for m in methods]
    ci_lowers = [bootstrap_results[m]["ci_lower"] for m in methods]
    ci_uppers = [bootstrap_results[m]["ci_upper"] for m in methods]
    errors = [[m - l for m, l in zip(means, ci_lowers)], [u - m for m, u in zip(means, ci_uppers)]]

    ax.bar(x_pos, means, color=[colors[m] for m in methods], alpha=0.7)
    ax.errorbar(x_pos, means, yerr=errors, fmt="none", color="black", capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Pareto Frontier Ratio")
    ax.set_title("Pareto Frontier Ratio with 95% Bootstrap CI")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figures_dir / "frontier_ratio_ci.png", dpi=150)
    plt.close()

    print(f"Saved visualizations to {figures_dir}")


def generate_report(results: dict, output_dir: Path) -> str:
    """Generate markdown report from results."""
    lines = [
        "# Statistical Analysis of GEPA Run Results",
        "",
        "## Summary",
        "",
        "This analysis examines whether observed Pareto frontier ratio differences",
        "between methods (adaboost, bayesian, baseline) are statistically significant",
        "or due to random variation.",
        "",
        "## Data Overview",
        "",
        "| Method | Candidates | Val Samples | Original Frontier Ratio |",
        "|--------|------------|-------------|------------------------|",
    ]

    for method in ["adaboost", "bayesian", "baseline"]:
        stats = results["quality"]["quality_stats"][method]
        original = results["original_frontier_ratios"][method]
        lines.append(f"| {method} | {stats['n_candidates']} | 30 | {original:.1%} |")

    lines.extend(
        [
            "",
            "## 1. Bootstrap Analysis of Frontier Ratios",
            "",
            "Bootstrap resampling (n=1000) on validation samples to estimate confidence intervals.",
            "",
            "| Method | Mean | 95% CI Lower | 95% CI Upper |",
            "|--------|------|--------------|--------------|",
        ]
    )

    for method in ["adaboost", "bayesian", "baseline"]:
        br = results["bootstrap"][method]
        lines.append(f"| {method} | {br['mean']:.1%} | {br['ci_lower']:.1%} | {br['ci_upper']:.1%} |")

    # Check CI overlap
    ci_overlap_ab = (
        results["bootstrap"]["adaboost"]["ci_lower"] <= results["bootstrap"]["bayesian"]["ci_upper"]
        and results["bootstrap"]["bayesian"]["ci_lower"] <= results["bootstrap"]["adaboost"]["ci_upper"]
    )
    ci_overlap_ba = (
        results["bootstrap"]["bayesian"]["ci_lower"] <= results["bootstrap"]["baseline"]["ci_upper"]
        and results["bootstrap"]["baseline"]["ci_lower"] <= results["bootstrap"]["bayesian"]["ci_upper"]
    )

    lines.extend(
        [
            "",
            f"**CI Overlap (adaboost vs bayesian):** {'Yes' if ci_overlap_ab else 'No'}",
            f"**CI Overlap (bayesian vs baseline):** {'Yes' if ci_overlap_ba else 'No'}",
            "",
            "## 2. Permutation Tests",
            "",
            "Testing H0: Method labels don't affect frontier ratio (n=10000 permutations).",
            "",
            "| Comparison | Observed Diff | p-value | Significant (α=0.05)? |",
            "|------------|---------------|---------|----------------------|",
        ]
    )

    for comparison in ["adaboost_vs_baseline", "bayesian_vs_baseline", "adaboost_vs_bayesian"]:
        pt = results["permutation_tests"][comparison]
        sig = "Yes" if pt["p_value"] < 0.05 else "No"
        lines.append(f"| {comparison} | {pt['observed_diff']:+.1%} | {pt['p_value']:.4f} | {sig} |")

    lines.extend(
        [
            "",
            "## 3. Candidate Quality Analysis",
            "",
            "Mean score per candidate (average across all validation samples).",
            "",
            "| Method | Mean | Std | Median | Min | Max |",
            "|--------|------|-----|--------|-----|-----|",
        ]
    )

    for method in ["adaboost", "bayesian", "baseline"]:
        qs = results["quality"]["quality_stats"][method]
        lines.append(f"| {method} | {qs['mean']:.3f} | {qs['std']:.3f} | {qs['median']:.3f} | {qs['min']:.3f} | {qs['max']:.3f} |")

    lines.extend(
        [
            "",
            "### Pairwise Statistical Tests",
            "",
            "| Comparison | Mann-Whitney U | p-value | Effect Size | KS Stat | KS p-value |",
            "|------------|----------------|---------|-------------|---------|------------|",
        ]
    )

    for comp, data in results["quality"]["pairwise_tests"].items():
        lines.append(
            f"| {comp} | {data['mann_whitney_u']:.1f} | {data['p_value']:.4f} | "
            f"{data['effect_size_rank_biserial']:.3f} | {data['ks_statistic']:.3f} | {data['ks_p_value']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Per-Sample Comparison",
            "",
            f"Sample wins (which method has highest success rate on each sample):",
            "",
        ]
    )

    for method, wins in results["per_sample"]["sample_wins"].items():
        lines.append(f"- **{method}**: {wins:.1f} samples")

    lines.extend(
        [
            "",
            "### Pairwise Sample-Level Comparisons",
            "",
        ]
    )

    for comp, data in results["per_sample"]["pairwise_comparisons"].items():
        m1, m2 = comp.replace("_vs_", " vs ").split(" vs ")
        lines.append(f"- **{comp}**: {m1} wins on {data['m1_wins']} samples, {m2} wins on {data['m2_wins']}, ties: {data['ties']}")

    lines.extend(
        [
            "",
            "## 5. Domination Structure",
            "",
            "Analysis of how candidates dominate each other within each method.",
            "",
            "| Method | Gini Coef | Max Dominator Ratio | Mean Dominates | Max Dominates |",
            "|--------|-----------|--------------------|-----------------|----|",
        ]
    )

    for method in ["adaboost", "bayesian", "baseline"]:
        dom = results["domination"][method]
        lines.append(
            f"| {method} | {dom['gini_coefficient']:.3f} | {dom['max_dominator_ratio']:.1%} | "
            f"{dom['dominates_count']['mean']:.1f} | {dom['dominates_count']['max']} |"
        )

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
        ]
    )

    # Generate conclusions based on results
    ada_vs_base_p = results["permutation_tests"]["adaboost_vs_baseline"]["p_value"]
    bay_vs_base_p = results["permutation_tests"]["bayesian_vs_baseline"]["p_value"]
    ada_vs_bay_p = results["permutation_tests"]["adaboost_vs_bayesian"]["p_value"]

    if bay_vs_base_p < 0.05:
        lines.append(
            f"1. **Bayesian vs Baseline**: The difference in frontier ratio is statistically significant "
            f"(p={bay_vs_base_p:.4f}). Bayesian method produces significantly more frontier candidates."
        )
    else:
        lines.append(
            f"1. **Bayesian vs Baseline**: The difference in frontier ratio is NOT statistically significant "
            f"(p={bay_vs_base_p:.4f}). The observed difference could be due to random variation."
        )

    if ada_vs_base_p < 0.05:
        lines.append(
            f"2. **Adaboost vs Baseline**: The difference is statistically significant (p={ada_vs_base_p:.4f})."
        )
    else:
        lines.append(
            f"2. **Adaboost vs Baseline**: The difference is NOT statistically significant (p={ada_vs_base_p:.4f})."
        )

    if ada_vs_bay_p < 0.05:
        lines.append(
            f"3. **Adaboost vs Bayesian**: The difference is statistically significant (p={ada_vs_bay_p:.4f})."
        )
    else:
        lines.append(
            f"3. **Adaboost vs Bayesian**: The difference is NOT statistically significant (p={ada_vs_bay_p:.4f})."
        )

    # Overall conclusion
    lines.extend(
        [
            "",
            "### Overall Assessment",
            "",
        ]
    )

    significant_diffs = sum([ada_vs_base_p < 0.05, bay_vs_base_p < 0.05, ada_vs_bay_p < 0.05])
    if significant_diffs == 0:
        lines.append(
            "**No statistically significant differences** were found between methods. "
            "The observed variations in Pareto frontier ratios are likely due to random variation "
            "and the limited sample size (30 validation samples, ~35 candidates per method)."
        )
    elif significant_diffs == 3:
        lines.append(
            "**All pairwise comparisons show significant differences.** The method choice "
            "appears to meaningfully affect the Pareto frontier structure."
        )
    else:
        lines.append(
            f"**{significant_diffs} of 3 pairwise comparisons show significant differences.** "
            "Some methods appear to differ, but not all comparisons reach significance."
        )

    lines.extend(
        [
            "",
            "---",
            "*Analysis generated by statistical_analysis_existing.py*",
        ]
    )

    report = "\n".join(lines)
    report_path = output_dir / "statistical_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to {report_path}")
    return report


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    data_dir = repo_root / "analysis_output"
    output_dir = data_dir

    print("=" * 60)
    print("Statistical Analysis of GEPA Run Results")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    scores_dict = load_scores(data_dir)

    # Compute original frontier ratios
    print("\n2. Computing original Pareto frontier ratios...")
    original_frontiers = {}
    for method, scores in scores_dict.items():
        frontier_indices, ratio = compute_pareto_frontier(scores)
        original_frontiers[method] = ratio
        print(f"  {method}: {ratio:.1%} ({len(frontier_indices)}/{scores.shape[0]} candidates)")

    # Bootstrap analysis
    print("\n3. Running bootstrap analysis (n=1000)...")
    bootstrap_results = {}
    for method, scores in scores_dict.items():
        mean, ci_lower, ci_upper, distribution = bootstrap_frontier_ratio(scores)
        bootstrap_results[method] = {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "distribution": distribution,
        }
        print(f"  {method}: {mean:.1%} [{ci_lower:.1%}, {ci_upper:.1%}]")

    # Permutation tests
    print("\n4. Running permutation tests (n=10000)...")
    permutation_results = {}
    comparisons = [
        ("adaboost", "baseline"),
        ("bayesian", "baseline"),
        ("adaboost", "bayesian"),
    ]
    for m1, m2 in comparisons:
        diff, p_value, _null_dist = permutation_test_frontier(scores_dict[m1], scores_dict[m2])
        permutation_results[f"{m1}_vs_{m2}"] = {
            "observed_diff": diff,
            "p_value": p_value,
        }
        print(f"  {m1} vs {m2}: diff={diff:+.1%}, p={p_value:.4f}")

    # Per-sample comparison
    print("\n5. Running per-sample comparison...")
    per_sample_results = per_sample_comparison(scores_dict)
    print(f"  Sample wins: {per_sample_results['sample_wins']}")

    # Candidate quality tests
    print("\n6. Running candidate quality tests...")
    quality_results = candidate_quality_test(scores_dict)
    for method, stats in quality_results["quality_stats"].items():
        print(f"  {method}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Domination analysis
    print("\n7. Running domination analysis...")
    domination_results = domination_analysis(scores_dict)
    for method, stats in domination_results.items():
        print(f"  {method}: gini={stats['gini_coefficient']:.3f}, max_dominator={stats['max_dominator_ratio']:.1%}")

    # Compile all results
    results = {
        "original_frontier_ratios": original_frontiers,
        "bootstrap": bootstrap_results,
        "permutation_tests": permutation_results,
        "per_sample": per_sample_results,
        "quality": quality_results,
        "domination": domination_results,
    }

    # Save results (excluding large distributions for JSON)
    results_for_json = {
        "original_frontier_ratios": original_frontiers,
        "bootstrap": {m: {k: v for k, v in d.items() if k != "distribution"} for m, d in bootstrap_results.items()},
        "permutation_tests": permutation_results,
        "per_sample": per_sample_results,
        "quality": quality_results,
        "domination": domination_results,
    }
    results_path = output_dir / "statistical_results.json"
    with open(results_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\n8. Results saved to {results_path}")

    # Create visualizations
    print("\n9. Creating visualizations...")
    create_visualizations(scores_dict, bootstrap_results, quality_results, output_dir)

    # Generate report
    print("\n10. Generating report...")
    generate_report(results, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
