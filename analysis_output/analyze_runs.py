#!/usr/bin/env python3
"""Analyze AdaBoost and Bayesian runs from wandb."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

OUTPUT_DIR = Path("/Users/panwenbo/Repos/gepa/analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Run configuration
ENTITY = "bmpixel"
PROJECT = "gepa-boost"
RUNS = {
    "adaboost": "nfadkixt",
    "bayesian": "k0jcelpb",
}


def download_run_tables(run_id: str, run_name: str) -> dict:
    """Download tables from a wandb run."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    result = {}

    # Get history data for time-series analysis
    history = list(run.scan_history(keys=[
        "gepa_iteration",
        "candidate/selected_idx",
        "candidate/new_idx",
        "candidate/best_idx",
        "train/batch_weight_avg",
        "train/weight_avg",
        "train/weight_max",
        "train/weight_min",
        "val/best_agg",
    ]))
    result["history"] = history

    # Download artifact tables
    for artifact in run.logged_artifacts():
        if "candidate_prompts" in artifact.name:
            try:
                table_path = artifact.download()
                table_file = os.path.join(table_path, "candidate_prompts.table.json")
                if os.path.exists(table_file):
                    with open(table_file, "r") as f:
                        result["candidate_prompts"] = json.load(f)
            except Exception as e:
                print(f"Error downloading candidate_prompts: {e}")

        if "train_sample_weights" in artifact.name:
            try:
                table_path = artifact.download()
                table_file = os.path.join(table_path, "train_sample_weights.table.json")
                if os.path.exists(table_file):
                    with open(table_file, "r") as f:
                        result["train_sample_weights"] = json.load(f)
            except Exception as e:
                print(f"Error downloading train_sample_weights: {e}")

        if "minibatch_outputs" in artifact.name:
            try:
                table_path = artifact.download()
                table_file = os.path.join(table_path, "minibatch_outputs.table.json")
                if os.path.exists(table_file):
                    with open(table_file, "r") as f:
                        result["minibatch_outputs"] = json.load(f)
            except Exception as e:
                print(f"Error downloading minibatch_outputs: {e}")

    return result


def parse_table(table_data: dict) -> pd.DataFrame:
    """Parse wandb table JSON to DataFrame."""
    columns = table_data.get("columns", [])
    data = table_data.get("data", [])
    return pd.DataFrame(data, columns=columns)


def analyze_candidate_genealogy(data: dict, run_name: str):
    """Analyze which parent candidates produced the best offspring."""
    print(f"\n{'='*60}")
    print(f"CANDIDATE GENEALOGY ANALYSIS - {run_name.upper()}")
    print(f"{'='*60}")

    if "candidate_prompts" not in data:
        print("No candidate_prompts table found")
        return None

    df = parse_table(data["candidate_prompts"])
    print(f"Columns: {list(df.columns)}")
    print(f"Total candidates: {len(df)}")

    # Column names may vary, let's check
    # Expected: candidate_idx, valset_aggregate_score, parent_idx, metric_calls_at_discovery, candidate_content

    # Find the score column
    score_col = None
    for col in df.columns:
        if "score" in col.lower() or "agg" in col.lower():
            score_col = col
            break

    # Find parent column
    parent_col = None
    for col in df.columns:
        if "parent" in col.lower():
            parent_col = col
            break

    # Find candidate index column
    idx_col = None
    for col in df.columns:
        if "idx" in col.lower() or "index" in col.lower():
            if "parent" not in col.lower():
                idx_col = col
                break

    if not all([score_col, parent_col, idx_col]):
        print(f"Could not find required columns. Found: {list(df.columns)}")
        return None

    print(f"\nUsing columns: idx={idx_col}, parent={parent_col}, score={score_col}")

    # Convert to numeric
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df[parent_col] = pd.to_numeric(df[parent_col], errors='coerce')
    df[idx_col] = pd.to_numeric(df[idx_col], errors='coerce')

    # Sort by score to find top candidates
    df_sorted = df.sort_values(score_col, ascending=False)

    print(f"\nTop 10 candidates by score:")
    for i, row in df_sorted.head(10).iterrows():
        print(f"  Candidate {int(row[idx_col])}: score={row[score_col]:.4f}, parent={int(row[parent_col]) if pd.notna(row[parent_col]) else 'None'}")

    # Analyze parent frequency for top candidates
    top_n = min(15, len(df) // 2)
    top_candidates = df_sorted.head(top_n)
    parent_counts = Counter(top_candidates[parent_col].dropna().astype(int))

    print(f"\nParent candidates that produced top {top_n} candidates:")
    for parent, count in parent_counts.most_common(10):
        parent_score = df[df[idx_col] == parent][score_col].values
        parent_score_str = f"{parent_score[0]:.4f}" if len(parent_score) > 0 else "N/A"
        print(f"  Parent {parent}: {count} children (parent score: {parent_score_str})")

    # Calculate concentration metric
    total_parents = len(parent_counts)
    top_parent_count = parent_counts.most_common(1)[0][1] if parent_counts else 0
    concentration = top_parent_count / top_n if top_n > 0 else 0

    print(f"\nConcentration analysis:")
    print(f"  Unique parents for top {top_n}: {total_parents}")
    print(f"  Top parent produced: {top_parent_count}/{top_n} = {concentration:.1%} of top candidates")

    if concentration > 0.3:
        print(f"  => Selection is CONCENTRATED (one parent dominates)")
    elif total_parents < top_n * 0.5:
        print(f"  => Selection is MODERATELY CONCENTRATED")
    else:
        print(f"  => Selection is DISPERSED (many different parents)")

    return {
        "df": df,
        "parent_counts": parent_counts,
        "concentration": concentration,
        "score_col": score_col,
        "parent_col": parent_col,
        "idx_col": idx_col,
    }


def analyze_candidate_selection_trend(data: dict, run_name: str):
    """Analyze the trend of candidate selection over iterations."""
    print(f"\n{'='*60}")
    print(f"CANDIDATE SELECTION TREND - {run_name.upper()}")
    print(f"{'='*60}")

    history = data.get("history", [])
    if not history:
        print("No history data found")
        return None

    # Extract selection data
    iterations = []
    selected_candidates = []
    best_candidates = []

    for entry in history:
        if entry.get("gepa_iteration") is not None and entry.get("candidate/selected_idx") is not None:
            iterations.append(entry["gepa_iteration"])
            selected_candidates.append(entry["candidate/selected_idx"])
            if entry.get("candidate/best_idx") is not None:
                best_candidates.append(entry["candidate/best_idx"])

    if not selected_candidates:
        print("No candidate selection data in history")
        return None

    print(f"Total iterations with selection data: {len(iterations)}")

    # Analyze selection concentration over time
    selection_counts = Counter(selected_candidates)
    print(f"\nMost frequently selected candidates:")
    for candidate, count in selection_counts.most_common(10):
        print(f"  Candidate {candidate}: selected {count} times ({count/len(selected_candidates):.1%})")

    # Divide into phases
    n = len(selected_candidates)
    early = selected_candidates[:n//3]
    mid = selected_candidates[n//3:2*n//3]
    late = selected_candidates[2*n//3:]

    print(f"\nSelection by phase:")
    print(f"  Early (iter 0-{n//3}): {len(set(early))} unique candidates, most common: {Counter(early).most_common(3)}")
    print(f"  Mid (iter {n//3}-{2*n//3}): {len(set(mid))} unique candidates, most common: {Counter(mid).most_common(3)}")
    print(f"  Late (iter {2*n//3}-{n}): {len(set(late))} unique candidates, most common: {Counter(late).most_common(3)}")

    return {
        "iterations": iterations,
        "selected_candidates": selected_candidates,
        "selection_counts": selection_counts,
    }


def analyze_sample_weights(data: dict, run_name: str):
    """Analyze training sample weight distribution."""
    print(f"\n{'='*60}")
    print(f"TRAINING SAMPLE WEIGHT ANALYSIS - {run_name.upper()}")
    print(f"{'='*60}")

    history = data.get("history", [])

    # Get weight statistics from history
    weight_stats = []
    for entry in history:
        if entry.get("train/weight_max") is not None:
            weight_stats.append({
                "iteration": entry.get("gepa_iteration", 0),
                "max": entry.get("train/weight_max", 0),
                "min": entry.get("train/weight_min", 0),
                "avg": entry.get("train/weight_avg", 0),
            })

    if weight_stats:
        print(f"\nWeight statistics from {len(weight_stats)} iterations:")

        # Early, mid, late analysis
        n = len(weight_stats)
        early = weight_stats[:n//3]
        mid = weight_stats[n//3:2*n//3]
        late = weight_stats[2*n//3:]

        def summarize_phase(phase, name):
            if not phase:
                return
            max_weights = [s["max"] for s in phase]
            min_weights = [s["min"] for s in phase]
            avg_weights = [s["avg"] for s in phase]

            max_range = max(max_weights) - min(min_weights) if max_weights else 0
            print(f"\n  {name} phase ({len(phase)} iterations):")
            print(f"    Max weight range: {min(min_weights):.3f} - {max(max_weights):.3f}")
            print(f"    Weight spread (max-min avg): {np.mean([s['max'] - s['min'] for s in phase]):.3f}")

            if max(max_weights) > 5:
                print(f"    => HIGHLY IMBALANCED (max > 5)")
            elif max(max_weights) > 2:
                print(f"    => MODERATELY IMBALANCED")
            else:
                print(f"    => RELATIVELY BALANCED")

        summarize_phase(early, "Early")
        summarize_phase(mid, "Mid")
        summarize_phase(late, "Late")

        # Overall trend
        if weight_stats:
            print(f"\n  Overall:")
            final_max = weight_stats[-1]["max"] if weight_stats else 0
            final_min = weight_stats[-1]["min"] if weight_stats else 0
            print(f"    Final iteration weights: min={final_min:.3f}, max={final_max:.3f}")
            print(f"    Weight ratio (max/min): {final_max/final_min:.1f}x" if final_min > 0 else "    Weight ratio: N/A")

    return weight_stats


def plot_candidate_genealogy(genealogy_data: dict, run_name: str):
    """Create visualization of candidate genealogy."""
    if genealogy_data is None:
        return

    df = genealogy_data["df"]
    score_col = genealogy_data["score_col"]
    parent_col = genealogy_data["parent_col"]
    idx_col = genealogy_data["idx_col"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score distribution
    ax1 = axes[0, 0]
    ax1.hist(df[score_col].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Validation Score")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Candidate Score Distribution - {run_name}")
    ax1.axvline(df[score_col].max(), color='red', linestyle='--', label=f'Best: {df[score_col].max():.3f}')
    ax1.legend()

    # 2. Parent frequency for all candidates
    ax2 = axes[0, 1]
    parent_counts = Counter(df[parent_col].dropna().astype(int))
    parents, counts = zip(*sorted(parent_counts.items())) if parent_counts else ([], [])
    ax2.bar(range(len(parents)), counts, tick_label=[str(int(p)) for p in parents])
    ax2.set_xlabel("Parent Candidate Index")
    ax2.set_ylabel("Number of Children")
    ax2.set_title(f"Children per Parent - {run_name}")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Score vs parent score
    ax3 = axes[1, 0]
    child_scores = []
    parent_scores = []
    for _, row in df.iterrows():
        if pd.notna(row[parent_col]):
            parent_idx = int(row[parent_col])
            parent_row = df[df[idx_col] == parent_idx]
            if len(parent_row) > 0:
                child_scores.append(row[score_col])
                parent_scores.append(parent_row[score_col].values[0])

    ax3.scatter(parent_scores, child_scores, alpha=0.6)
    ax3.plot([0, 1], [0, 1], 'r--', label='y=x (no improvement)')
    ax3.set_xlabel("Parent Score")
    ax3.set_ylabel("Child Score")
    ax3.set_title(f"Parent vs Child Score - {run_name}")
    ax3.legend()

    # 4. Score progression
    ax4 = axes[1, 1]
    ax4.plot(df[idx_col], df[score_col], 'o-', alpha=0.6)
    ax4.set_xlabel("Candidate Index (chronological)")
    ax4.set_ylabel("Validation Score")
    ax4.set_title(f"Score Progression - {run_name}")

    # Add rolling max
    df_sorted_idx = df.sort_values(idx_col)
    rolling_max = df_sorted_idx[score_col].cummax()
    ax4.plot(df_sorted_idx[idx_col], rolling_max, 'r-', linewidth=2, label='Best so far')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"candidate_genealogy_{run_name}.png", dpi=150)
    plt.close()
    print(f"Saved: candidate_genealogy_{run_name}.png")


def plot_selection_trend(selection_data: dict, run_name: str):
    """Create visualization of candidate selection over time."""
    if selection_data is None:
        return

    iterations = selection_data["iterations"]
    selected = selection_data["selected_candidates"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Selection over time
    ax1 = axes[0]
    ax1.scatter(iterations, selected, alpha=0.5, s=10)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Selected Candidate Index")
    ax1.set_title(f"Candidate Selection Over Time - {run_name}")

    # 2. Selection frequency heatmap
    ax2 = axes[1]
    selection_counts = selection_data["selection_counts"]
    candidates, counts = zip(*sorted(selection_counts.items())) if selection_counts else ([], [])
    ax2.bar(range(len(candidates)), counts, tick_label=[str(int(c)) for c in candidates])
    ax2.set_xlabel("Candidate Index")
    ax2.set_ylabel("Times Selected")
    ax2.set_title(f"Selection Frequency - {run_name}")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"selection_trend_{run_name}.png", dpi=150)
    plt.close()
    print(f"Saved: selection_trend_{run_name}.png")


def plot_weight_evolution(weight_stats: list, run_name: str):
    """Plot weight evolution over iterations."""
    if not weight_stats:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = [s["iteration"] for s in weight_stats]
    max_weights = [s["max"] for s in weight_stats]
    min_weights = [s["min"] for s in weight_stats]
    avg_weights = [s["avg"] for s in weight_stats]

    ax.fill_between(iterations, min_weights, max_weights, alpha=0.3, label='Weight range')
    ax.plot(iterations, avg_weights, 'b-', linewidth=2, label='Average weight')
    ax.plot(iterations, max_weights, 'r--', alpha=0.7, label='Max weight')
    ax.plot(iterations, min_weights, 'g--', alpha=0.7, label='Min weight')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sample Weight")
    ax.set_title(f"Sample Weight Evolution - {run_name}")
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"weight_evolution_{run_name}.png", dpi=150)
    plt.close()
    print(f"Saved: weight_evolution_{run_name}.png")


def compare_runs(results: dict):
    """Compare the two runs side by side."""
    print(f"\n{'='*60}")
    print("COMPARISON: ADABOOST vs BAYESIAN")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Selection concentration comparison
    ax1 = axes[0, 0]
    concentrations = []
    for run_name in ["adaboost", "bayesian"]:
        if run_name in results and results[run_name].get("selection"):
            counts = results[run_name]["selection"]["selection_counts"]
            total = sum(counts.values())
            top_count = counts.most_common(1)[0][1] if counts else 0
            concentrations.append(top_count / total if total > 0 else 0)
        else:
            concentrations.append(0)

    ax1.bar(["AdaBoost", "Bayesian"], concentrations, color=['blue', 'green'])
    ax1.set_ylabel("Selection Concentration")
    ax1.set_title("Most Selected Candidate / Total Selections")

    # 2. Weight range comparison
    ax2 = axes[0, 1]
    for run_name, color in [("adaboost", "blue"), ("bayesian", "green")]:
        if run_name in results and results[run_name].get("weights"):
            stats = results[run_name]["weights"]
            if stats:
                iterations = [s["iteration"] for s in stats]
                max_weights = [s["max"] for s in stats]
                ax2.plot(iterations, max_weights, color=color, label=run_name.title())

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Max Weight")
    ax2.set_title("Maximum Sample Weight Over Time")
    ax2.legend()
    ax2.set_yscale('log')

    # 3. Unique parents comparison
    ax3 = axes[1, 0]
    unique_parents = []
    for run_name in ["adaboost", "bayesian"]:
        if run_name in results and results[run_name].get("genealogy"):
            parent_counts = results[run_name]["genealogy"]["parent_counts"]
            unique_parents.append(len(parent_counts))
        else:
            unique_parents.append(0)

    ax3.bar(["AdaBoost", "Bayesian"], unique_parents, color=['blue', 'green'])
    ax3.set_ylabel("Number of Unique Parents")
    ax3.set_title("Diversity of Parent Selection")

    # 4. Best score progression comparison
    ax4 = axes[1, 1]
    for run_name, color in [("adaboost", "blue"), ("bayesian", "green")]:
        if run_name in results and results[run_name].get("genealogy"):
            df = results[run_name]["genealogy"]["df"]
            score_col = results[run_name]["genealogy"]["score_col"]
            idx_col = results[run_name]["genealogy"]["idx_col"]

            df_sorted = df.sort_values(idx_col)
            rolling_max = df_sorted[score_col].cummax()
            ax4.plot(df_sorted[idx_col], rolling_max, color=color, label=run_name.title(), linewidth=2)

    ax4.set_xlabel("Candidate Index")
    ax4.set_ylabel("Best Score So Far")
    ax4.set_title("Best Score Progression")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"Saved: comparison.png")


def main():
    """Main analysis function."""
    results = {}

    for run_name, run_id in RUNS.items():
        print(f"\n{'#'*70}")
        print(f"ANALYZING {run_name.upper()} RUN ({run_id})")
        print(f"{'#'*70}")

        print(f"\nDownloading data from wandb...")
        data = download_run_tables(run_id, run_name)

        # Analyze candidate genealogy
        genealogy = analyze_candidate_genealogy(data, run_name)

        # Analyze selection trend
        selection = analyze_candidate_selection_trend(data, run_name)

        # Analyze sample weights
        weights = analyze_sample_weights(data, run_name)

        # Create visualizations
        plot_candidate_genealogy(genealogy, run_name)
        plot_selection_trend(selection, run_name)
        plot_weight_evolution(weights, run_name)

        results[run_name] = {
            "data": data,
            "genealogy": genealogy,
            "selection": selection,
            "weights": weights,
        }

    # Compare runs
    compare_runs(results)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
