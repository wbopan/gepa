#!/usr/bin/env python3
"""Detailed analysis of sample selection patterns for AdaBoost and Bayesian runs."""

import json
import os
from collections import Counter, defaultdict
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


def get_full_history(run_id: str) -> pd.DataFrame:
    """Get full history from a wandb run."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Get all history
    history = run.history(samples=10000)  # Get all samples
    return history


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
                        # Extract version from artifact name to get iteration
                        version = artifact.version
                        df["artifact_version"] = version
                        weight_tables.append(df)
            except Exception as e:
                print(f"Error: {e}")

    return weight_tables


def get_all_minibatch_tables(run_id: str) -> list[pd.DataFrame]:
    """Get all minibatch_outputs tables."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    minibatch_tables = []
    for artifact in run.logged_artifacts():
        if "minibatch_outputs" in artifact.name:
            try:
                table_path = artifact.download()
                table_file = os.path.join(table_path, "minibatch_outputs.table.json")
                if os.path.exists(table_file):
                    with open(table_file, "r") as f:
                        data = json.load(f)
                        columns = data.get("columns", [])
                        rows = data.get("data", [])
                        df = pd.DataFrame(rows, columns=columns)
                        version = artifact.version
                        df["artifact_version"] = version
                        minibatch_tables.append(df)
            except Exception as e:
                print(f"Error: {e}")

    return minibatch_tables


def analyze_history_metrics(run_id: str, run_name: str):
    """Analyze metrics from run history."""
    print(f"\n{'='*60}")
    print(f"HISTORY METRICS - {run_name.upper()}")
    print(f"{'='*60}")

    history = get_full_history(run_id)
    print(f"History shape: {history.shape}")
    print(f"Columns: {list(history.columns)[:20]}...")  # First 20 columns

    # Find relevant columns
    weight_cols = [c for c in history.columns if 'weight' in c.lower()]
    candidate_cols = [c for c in history.columns if 'candidate' in c.lower()]

    print(f"\nWeight-related columns: {weight_cols}")
    print(f"Candidate-related columns: {candidate_cols}")

    return history


def analyze_sample_selection_from_minibatch(minibatch_tables: list[pd.DataFrame], run_name: str):
    """Analyze which samples are selected in minibatches."""
    print(f"\n{'='*60}")
    print(f"SAMPLE SELECTION ANALYSIS - {run_name.upper()}")
    print(f"{'='*60}")

    if not minibatch_tables:
        print("No minibatch tables found")
        return None

    # Combine all tables
    all_rows = []
    for i, df in enumerate(minibatch_tables):
        df = df.copy()
        df["table_idx"] = i
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total minibatch entries: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")

    # Find sample ID column
    sample_col = None
    for col in combined.columns:
        if "sample" in col.lower() or "train" in col.lower():
            if "id" in col.lower() or "idx" in col.lower():
                sample_col = col
                break

    if sample_col is None:
        # Try to find iteration column
        if "iteration" in combined.columns:
            print("Using iteration column to infer sample distribution")
        print("Available columns:", list(combined.columns))
        return None

    print(f"Using sample column: {sample_col}")

    # Count sample occurrences
    sample_counts = Counter(combined[sample_col])

    print(f"\nSample selection frequency:")
    for sample_id, count in sample_counts.most_common(10):
        print(f"  Sample {sample_id}: selected {count} times")

    # Analyze by phases (using table_idx as proxy for iteration)
    n_tables = len(minibatch_tables)
    early_tables = [t for i, t in enumerate(minibatch_tables) if i < n_tables // 3]
    mid_tables = [t for i, t in enumerate(minibatch_tables) if n_tables // 3 <= i < 2 * n_tables // 3]
    late_tables = [t for i, t in enumerate(minibatch_tables) if i >= 2 * n_tables // 3]

    def phase_stats(tables, name):
        if not tables:
            return
        df = pd.concat(tables, ignore_index=True)
        counts = Counter(df[sample_col])
        print(f"\n  {name} phase ({len(tables)} tables):")
        print(f"    Unique samples: {len(counts)}")
        print(f"    Most common: {counts.most_common(3)}")

    phase_stats(early_tables, "Early")
    phase_stats(mid_tables, "Mid")
    phase_stats(late_tables, "Late")

    return sample_counts


def analyze_weight_distribution(weight_tables: list[pd.DataFrame], run_name: str):
    """Analyze weight distribution across iterations."""
    print(f"\n{'='*60}")
    print(f"WEIGHT DISTRIBUTION - {run_name.upper()}")
    print(f"{'='*60}")

    if not weight_tables:
        print("No weight tables found")
        return None

    print(f"Found {len(weight_tables)} weight tables")

    # Sample some tables
    n = len(weight_tables)

    # Get early, mid, late tables
    early_idx = n // 6 if n > 6 else 0
    mid_idx = n // 2
    late_idx = n - 1

    def analyze_table(df, name):
        print(f"\n{name}:")
        print(f"  Columns: {list(df.columns)}")

        # Find weight column
        weight_col = None
        for col in df.columns:
            if col.lower() == "weight" or "weight" in col.lower():
                weight_col = col
                break

        if weight_col is None:
            print("  No weight column found")
            return None

        weights = pd.to_numeric(df[weight_col], errors='coerce').dropna()
        print(f"  Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}, std={weights.std():.4f}")
        print(f"  Weight ratio (max/min): {weights.max()/weights.min():.1f}x" if weights.min() > 0 else "  Ratio: N/A")

        # Gini coefficient to measure inequality
        sorted_weights = np.sort(weights.values)
        n = len(sorted_weights)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_weights) / (n * np.sum(sorted_weights))) - (n + 1) / n
        print(f"  Gini coefficient: {gini:.4f} (0=equal, 1=max inequality)")

        return weights

    early_weights = analyze_table(weight_tables[early_idx], f"Early (table {early_idx})")
    mid_weights = analyze_table(weight_tables[mid_idx], f"Mid (table {mid_idx})")
    late_weights = analyze_table(weight_tables[late_idx], f"Late (table {late_idx})")

    return {"early": early_weights, "mid": mid_weights, "late": late_weights}


def analyze_consistently_selected_samples(weight_tables: list[pd.DataFrame], run_name: str):
    """Find samples that are consistently selected with high weights."""
    print(f"\n{'='*60}")
    print(f"CONSISTENTLY HIGH-WEIGHT SAMPLES - {run_name.upper()}")
    print(f"{'='*60}")

    if len(weight_tables) < 5:
        print("Not enough weight tables for analysis")
        return

    # Get sample ID and weight columns
    sample_col = None
    weight_col = None

    for col in weight_tables[0].columns:
        if "sample" in col.lower() or "id" in col.lower():
            if "weight" not in col.lower():
                sample_col = col
        if col.lower() == "weight" or "weight" in col.lower():
            weight_col = col

    if sample_col is None or weight_col is None:
        print(f"Could not find sample/weight columns. Columns: {list(weight_tables[0].columns)}")
        return

    print(f"Using columns: sample={sample_col}, weight={weight_col}")

    # Track weight history per sample
    sample_weights_history = defaultdict(list)

    for df in weight_tables:
        for _, row in df.iterrows():
            sample_id = row.get(sample_col)
            weight = row.get(weight_col)
            if sample_id is not None and weight is not None:
                sample_weights_history[sample_id].append(float(weight))

    # Find samples with consistently high weights
    print(f"\nSamples with consistently high weights (top 10 by average weight):")

    avg_weights = {}
    for sample_id, weights in sample_weights_history.items():
        if len(weights) >= len(weight_tables) // 2:  # At least half the iterations
            avg_weights[sample_id] = np.mean(weights)

    sorted_samples = sorted(avg_weights.items(), key=lambda x: -x[1])
    for sample_id, avg_w in sorted_samples[:10]:
        weights = sample_weights_history[sample_id]
        print(f"  Sample {sample_id}: avg={avg_w:.3f}, min={min(weights):.3f}, max={max(weights):.3f}, count={len(weights)}")

    # Find samples with consistently low weights (easy samples)
    print(f"\nSamples with consistently low weights (top 10 by lowest average):")
    for sample_id, avg_w in sorted_samples[-10:]:
        weights = sample_weights_history[sample_id]
        print(f"  Sample {sample_id}: avg={avg_w:.3f}, min={min(weights):.3f}, max={max(weights):.3f}, count={len(weights)}")

    return sample_weights_history


def plot_weight_evolution_detailed(weight_tables: list[pd.DataFrame], run_name: str):
    """Plot detailed weight evolution."""
    if len(weight_tables) < 3:
        return

    sample_col = None
    weight_col = None

    for col in weight_tables[0].columns:
        if "sample" in col.lower() or "id" in col.lower():
            if "weight" not in col.lower():
                sample_col = col
        if col.lower() == "weight" or "weight" in col.lower():
            weight_col = col

    if sample_col is None or weight_col is None:
        return

    # Track weights over time
    sample_ids = set()
    for df in weight_tables[:10]:  # Sample first 10 tables
        for _, row in df.iterrows():
            sample_ids.add(row.get(sample_col))

    sample_ids = list(sample_ids)[:30]  # Limit to 30 samples

    # Build time series
    weight_series = {sid: [] for sid in sample_ids}
    for df in weight_tables:
        sample_weights = dict(zip(df[sample_col], df[weight_col]))
        for sid in sample_ids:
            weight_series[sid].append(sample_weights.get(sid, np.nan))

    fig, ax = plt.subplots(figsize=(14, 8))

    for sid in sample_ids:
        ax.plot(weight_series[sid], alpha=0.5, linewidth=1)

    ax.set_xlabel("Iteration (table index)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Individual Sample Weight Evolution - {run_name}")
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weight_evolution_detailed_{run_name}.png", dpi=150)
    plt.close()
    print(f"Saved: weight_evolution_detailed_{run_name}.png")


def plot_weight_distribution_comparison(adaboost_weights: dict, bayesian_weights: dict):
    """Compare weight distributions between the two methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    phases = ["early", "mid", "late"]

    for i, phase in enumerate(phases):
        ax = axes[0, i]
        if adaboost_weights and adaboost_weights.get(phase) is not None:
            w = adaboost_weights[phase]
            ax.hist(w, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title(f"AdaBoost - {phase.capitalize()}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")

        ax = axes[1, i]
        if bayesian_weights and bayesian_weights.get(phase) is not None:
            w = bayesian_weights[phase]
            ax.hist(w, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.set_title(f"Bayesian - {phase.capitalize()}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weight_distribution_comparison.png", dpi=150)
    plt.close()
    print("Saved: weight_distribution_comparison.png")


def main():
    results = {}

    for run_name, run_id in RUNS.items():
        print(f"\n{'#'*70}")
        print(f"DETAILED ANALYSIS: {run_name.upper()} ({run_id})")
        print(f"{'#'*70}")

        # Get history
        history = analyze_history_metrics(run_id, run_name)

        # Get weight tables
        print(f"\nDownloading weight tables...")
        weight_tables = get_all_weight_tables(run_id)
        print(f"Downloaded {len(weight_tables)} weight tables")

        # Get minibatch tables
        print(f"\nDownloading minibatch tables...")
        minibatch_tables = get_all_minibatch_tables(run_id)
        print(f"Downloaded {len(minibatch_tables)} minibatch tables")

        # Analyze
        sample_counts = analyze_sample_selection_from_minibatch(minibatch_tables, run_name)
        weight_dist = analyze_weight_distribution(weight_tables, run_name)
        sample_weight_history = analyze_consistently_selected_samples(weight_tables, run_name)

        # Plot
        plot_weight_evolution_detailed(weight_tables, run_name)

        results[run_name] = {
            "history": history,
            "weight_tables": weight_tables,
            "minibatch_tables": minibatch_tables,
            "sample_counts": sample_counts,
            "weight_dist": weight_dist,
            "sample_weight_history": sample_weight_history,
        }

    # Comparison plots
    if results.get("adaboost") and results.get("bayesian"):
        plot_weight_distribution_comparison(
            results["adaboost"].get("weight_dist"),
            results["bayesian"].get("weight_dist")
        )

    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
