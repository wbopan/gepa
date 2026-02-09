"""Analyze adaboost run mutations to answer key questions about evolution dynamics."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# Configuration
ENTITY = "bmpixel"
PROJECT = "gepa-boost"
RUN_ID = "nfadkixt"  # adaboost run
RUN_NAME = "adaboost"


def fetch_all_data():
    """Fetch all relevant data from the wandb run."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    print(f"Fetching data for {RUN_NAME} ({RUN_ID})...")
    print(f"Run state: {run.state}")
    print(f"Run summary keys: {list(run.summary.keys())[:20]}...")

    data = {}

    # Fetch history
    print("\n--- Fetching History ---")
    history = run.history(
        keys=[
            "gepa_iteration",
            "val/new_score",
            "val/best_agg",
            "candidate/selected_idx",
            "candidate/new_idx",
            "train/batch_score_before",
            "train/batch_score_after",
            "train/batch_weight_avg",
        ],
        pandas=True,
    )
    print(f"History rows: {len(history)}")
    print(f"History columns: {list(history.columns)}")
    data["history"] = history

    # Fetch artifacts
    print("\n--- Fetching Artifacts ---")
    artifacts = list(run.logged_artifacts())
    print(f"Total artifacts: {len(artifacts)}")

    # Group artifacts by type
    artifact_groups = {}
    for art in artifacts:
        base_name = art.name.split(":")[0]
        if base_name not in artifact_groups:
            artifact_groups[base_name] = []
        artifact_groups[base_name].append(art)

    print(f"Artifact types: {list(artifact_groups.keys())}")

    # Fetch minibatch_outputs (get multiple versions to see evolution)
    minibatch_arts = artifact_groups.get("run-nfadkixt-minibatch_outputs", [])
    print(f"\nMinibatch_outputs artifacts: {len(minibatch_arts)}")

    all_minibatch_rows = []
    # Sample every 10th artifact to avoid too much data
    for art in minibatch_arts[::10]:
        try:
            table = art.get("minibatch_outputs")
            if table:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                all_minibatch_rows.append(df)
        except Exception as e:
            print(f"  Error loading {art.name}: {e}")
            continue

    if all_minibatch_rows:
        data["minibatch_outputs"] = pd.concat(all_minibatch_rows, ignore_index=True)
        print(f"Total minibatch rows: {len(data['minibatch_outputs'])}")
    else:
        # Try getting the latest
        if minibatch_arts:
            latest_art = minibatch_arts[-1]
            try:
                table = latest_art.get("minibatch_outputs")
                if table:
                    data["minibatch_outputs"] = pd.DataFrame(data=table.data, columns=table.columns)
                    print(f"Loaded latest minibatch: {len(data['minibatch_outputs'])} rows")
            except Exception as e:
                print(f"Error loading latest minibatch: {e}")

    # Fetch candidate_prompts (latest)
    candidate_arts = artifact_groups.get("run-nfadkixt-candidate_prompts", [])
    if candidate_arts:
        latest_art = candidate_arts[-1]
        try:
            table = latest_art.get("candidate_prompts")
            if table:
                data["candidate_prompts"] = pd.DataFrame(data=table.data, columns=table.columns)
                print(f"Candidate prompts: {len(data['candidate_prompts'])} rows")
                print(f"Columns: {list(data['candidate_prompts'].columns)}")
        except Exception as e:
            print(f"Error loading candidate_prompts: {e}")

    # Fetch valset_outputs (latest)
    valset_arts = artifact_groups.get("run-nfadkixt-valset_outputs", [])
    if valset_arts:
        latest_art = valset_arts[-1]
        try:
            table = latest_art.get("valset_outputs")
            if table:
                data["valset_outputs"] = pd.DataFrame(data=table.data, columns=table.columns)
                print(f"Valset outputs: {len(data['valset_outputs'])} rows")
        except Exception as e:
            print(f"Error loading valset_outputs: {e}")

    return data


def analyze_score_progression(data: dict):
    """Q1: Is the average proposed candidate score increasing?"""
    print("\n" + "=" * 70)
    print("Q1: Is the average proposed candidate score increasing over time?")
    print("=" * 70)

    history = data["history"]

    if "val/new_score" not in history.columns:
        print("val/new_score not in history")
        return

    # Get non-null scores with their iterations
    scores_df = history[["gepa_iteration", "val/new_score"]].dropna()
    if scores_df.empty:
        print("No val/new_score data found")
        return

    scores_df = scores_df.sort_values("gepa_iteration")
    print(f"Total new candidate scores: {len(scores_df)}")

    # Split into early, mid, late phases
    n = len(scores_df)
    early = scores_df.iloc[: n // 3]
    mid = scores_df.iloc[n // 3 : 2 * n // 3]
    late = scores_df.iloc[2 * n // 3 :]

    print(f"\nScore progression by phase:")
    print(f"  Early (iter {early['gepa_iteration'].min():.0f}-{early['gepa_iteration'].max():.0f}):")
    print(f"    Mean: {early['val/new_score'].mean():.4f}")
    print(f"    Std:  {early['val/new_score'].std():.4f}")
    print(f"    Range: [{early['val/new_score'].min():.4f}, {early['val/new_score'].max():.4f}]")

    print(f"  Mid (iter {mid['gepa_iteration'].min():.0f}-{mid['gepa_iteration'].max():.0f}):")
    print(f"    Mean: {mid['val/new_score'].mean():.4f}")
    print(f"    Std:  {mid['val/new_score'].std():.4f}")
    print(f"    Range: [{mid['val/new_score'].min():.4f}, {mid['val/new_score'].max():.4f}]")

    print(f"  Late (iter {late['gepa_iteration'].min():.0f}-{late['gepa_iteration'].max():.0f}):")
    print(f"    Mean: {late['val/new_score'].mean():.4f}")
    print(f"    Std:  {late['val/new_score'].std():.4f}")
    print(f"    Range: [{late['val/new_score'].min():.4f}, {late['val/new_score'].max():.4f}]")

    # Check trend
    from scipy import stats

    iterations = scores_df["gepa_iteration"].values
    scores = scores_df["val/new_score"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, scores)

    print(f"\nLinear trend analysis:")
    print(f"  Slope: {slope:.6f} (per iteration)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if slope > 0 and p_value < 0.05:
        print("  -> Significant positive trend: scores ARE increasing")
    elif slope < 0 and p_value < 0.05:
        print("  -> Significant negative trend: scores are DECREASING")
    else:
        print("  -> No significant trend")


def analyze_minibatch_variation(data: dict):
    """Q2: How much variation is there in mini-batch samples?"""
    print("\n" + "=" * 70)
    print("Q2: How much variation is there in mini-batch samples?")
    print("=" * 70)

    if "minibatch_outputs" not in data:
        print("No minibatch_outputs data available")
        return

    mb = data["minibatch_outputs"]
    print(f"Columns: {list(mb.columns)}")
    print(f"Total rows: {len(mb)}")

    # Group by iteration to see sample IDs per iteration
    if "iteration" in mb.columns and "train_sample_id" in mb.columns:
        sample_sets_by_iter = mb.groupby("iteration")["train_sample_id"].apply(set).to_dict()

        print(f"\nUnique iterations: {len(sample_sets_by_iter)}")

        # Calculate overlap between consecutive iterations
        iterations = sorted(sample_sets_by_iter.keys())
        overlaps = []
        for i in range(1, len(iterations)):
            prev_iter = iterations[i - 1]
            curr_iter = iterations[i]
            prev_samples = sample_sets_by_iter[prev_iter]
            curr_samples = sample_sets_by_iter[curr_iter]
            if prev_samples and curr_samples:
                overlap = len(prev_samples & curr_samples) / len(prev_samples | curr_samples)
                overlaps.append(overlap)

        if overlaps:
            print(f"\nSample overlap between consecutive iterations:")
            print(f"  Mean Jaccard similarity: {np.mean(overlaps):.4f}")
            print(f"  Std: {np.std(overlaps):.4f}")
            print(f"  Min: {np.min(overlaps):.4f}")
            print(f"  Max: {np.max(overlaps):.4f}")

        # Count how often each sample ID appears
        all_sample_ids = mb["train_sample_id"].tolist()
        sample_counts = Counter(all_sample_ids)
        counts = list(sample_counts.values())

        print(f"\nSample frequency distribution:")
        print(f"  Unique samples: {len(sample_counts)}")
        print(f"  Mean appearances: {np.mean(counts):.2f}")
        print(f"  Std appearances: {np.std(counts):.2f}")
        print(f"  Min appearances: {min(counts)}")
        print(f"  Max appearances: {max(counts)}")

        # Show most/least frequent samples
        print(f"\n  Most frequent samples: {sample_counts.most_common(5)}")
        print(f"  Least frequent samples: {sample_counts.most_common()[-5:]}")


def analyze_parent_child_comparison(data: dict):
    """Q3: Do children score higher than parents? Compare early vs late."""
    print("\n" + "=" * 70)
    print("Q3: Do children score higher than parents? (Early vs Late)")
    print("=" * 70)

    if "candidate_prompts" not in data:
        print("No candidate_prompts data available")
        return

    cp = data["candidate_prompts"]
    print(f"Columns: {list(cp.columns)}")

    # Identify score column
    score_col = None
    for col in cp.columns:
        if "score" in col.lower():
            score_col = col
            break

    if score_col is None:
        print("No score column found")
        return

    print(f"Using score column: {score_col}")

    # Build parent-child relationships
    improvements = []
    for idx, row in cp.iterrows():
        parent_idx = row.get("parent_idx")
        child_score = row.get(score_col)
        child_idx = row.get("candidate_idx")

        if parent_idx is not None and not (isinstance(parent_idx, float) and np.isnan(parent_idx)):
            try:
                parent_idx = int(parent_idx)
                parent_row = cp[cp["candidate_idx"] == parent_idx]
                if not parent_row.empty:
                    parent_score = parent_row.iloc[0][score_col]
                    if not np.isnan(parent_score) and not np.isnan(child_score):
                        improvements.append(
                            {
                                "child_idx": child_idx,
                                "parent_idx": parent_idx,
                                "parent_score": parent_score,
                                "child_score": child_score,
                                "improvement": child_score - parent_score,
                            }
                        )
            except (ValueError, TypeError):
                continue

    if not improvements:
        print("No parent-child relationships found")
        return

    imp_df = pd.DataFrame(improvements)
    print(f"\nTotal parent-child pairs: {len(imp_df)}")

    # Overall statistics
    print(f"\nOverall improvement statistics:")
    print(f"  Mean improvement: {imp_df['improvement'].mean():.4f}")
    print(f"  Median improvement: {imp_df['improvement'].median():.4f}")
    print(f"  Std: {imp_df['improvement'].std():.4f}")
    print(f"  Children better than parent: {(imp_df['improvement'] > 0).sum()} ({100*(imp_df['improvement'] > 0).mean():.1f}%)")
    print(f"  Children equal to parent: {(imp_df['improvement'] == 0).sum()} ({100*(imp_df['improvement'] == 0).mean():.1f}%)")
    print(f"  Children worse than parent: {(imp_df['improvement'] < 0).sum()} ({100*(imp_df['improvement'] < 0).mean():.1f}%)")

    # Split into early and late (by child_idx as proxy for time)
    n = len(imp_df)
    early_df = imp_df.iloc[: n // 2]
    late_df = imp_df.iloc[n // 2 :]

    print(f"\nEarly phase (first {len(early_df)} mutations):")
    print(f"  Mean improvement: {early_df['improvement'].mean():.4f}")
    print(f"  Children better: {(early_df['improvement'] > 0).sum()} ({100*(early_df['improvement'] > 0).mean():.1f}%)")
    print(f"  Mean child score: {early_df['child_score'].mean():.4f}")
    print(f"  Mean parent score: {early_df['parent_score'].mean():.4f}")

    print(f"\nLate phase (last {len(late_df)} mutations):")
    print(f"  Mean improvement: {late_df['improvement'].mean():.4f}")
    print(f"  Children better: {(late_df['improvement'] > 0).sum()} ({100*(late_df['improvement'] > 0).mean():.1f}%)")
    print(f"  Mean child score: {late_df['child_score'].mean():.4f}")
    print(f"  Mean parent score: {late_df['parent_score'].mean():.4f}")


def analyze_sample_level_similarity(data: dict):
    """Q4: Are child scores similar to parent scores at sample level?"""
    print("\n" + "=" * 70)
    print("Q4: Sample-level similarity between parent and child scores")
    print("=" * 70)

    if "minibatch_outputs" not in data:
        print("No minibatch_outputs data available")
        return

    mb = data["minibatch_outputs"]

    if "score_before" not in mb.columns or "score_after" not in mb.columns:
        print(f"Required columns not found. Available: {list(mb.columns)}")
        return

    # Filter out rows with missing scores
    valid_mb = mb.dropna(subset=["score_before", "score_after"])
    print(f"Valid samples with both scores: {len(valid_mb)}")

    # Per-sample analysis
    print(f"\nPer-sample score comparison (parent vs child on same mini-batch sample):")
    score_before = valid_mb["score_before"].values
    score_after = valid_mb["score_after"].values

    same = np.sum(score_before == score_after)
    improved = np.sum(score_after > score_before)
    degraded = np.sum(score_after < score_before)

    print(f"  Same score: {same} ({100*same/len(valid_mb):.1f}%)")
    print(f"  Improved (child > parent): {improved} ({100*improved/len(valid_mb):.1f}%)")
    print(f"  Degraded (child < parent): {degraded} ({100*degraded/len(valid_mb):.1f}%)")

    # Correlation
    from scipy import stats

    if len(score_before) > 10:
        corr, p_value = stats.pearsonr(score_before, score_after)
        print(f"\n  Pearson correlation: {corr:.4f} (p={p_value:.4f})")

    # Agreement rate (both correct or both wrong, assuming binary scores)
    if set(score_before).issubset({0, 1, 0.0, 1.0}) and set(score_after).issubset({0, 1, 0.0, 1.0}):
        agreement = np.mean(score_before == score_after)
        print(f"  Agreement rate (binary): {agreement:.4f}")

    # Per-iteration analysis
    if "iteration" in valid_mb.columns:
        print(f"\nPer-iteration analysis:")
        iter_stats = []
        for iteration, group in valid_mb.groupby("iteration"):
            sb = group["score_before"].values
            sa = group["score_after"].values
            improved_pct = np.mean(sa > sb)
            same_pct = np.mean(sa == sb)
            degraded_pct = np.mean(sa < sb)
            iter_stats.append(
                {"iteration": iteration, "improved_pct": improved_pct, "same_pct": same_pct, "degraded_pct": degraded_pct}
            )

        iter_df = pd.DataFrame(iter_stats)
        print(f"  Across {len(iter_df)} iterations:")
        print(f"    Mean improved %: {100*iter_df['improved_pct'].mean():.1f}%")
        print(f"    Mean same %: {100*iter_df['same_pct'].mean():.1f}%")
        print(f"    Mean degraded %: {100*iter_df['degraded_pct'].mean():.1f}%")


def analyze_batch_scores(data: dict):
    """Additional: Analyze batch score changes (sum of scores on mini-batch)."""
    print("\n" + "=" * 70)
    print("Additional: Batch-level score comparison")
    print("=" * 70)

    history = data["history"]

    if "train/batch_score_before" in history.columns and "train/batch_score_after" in history.columns:
        batch_df = history[["gepa_iteration", "train/batch_score_before", "train/batch_score_after"]].dropna()
        print(f"Batch score records: {len(batch_df)}")

        before = batch_df["train/batch_score_before"].values
        after = batch_df["train/batch_score_after"].values

        improved = np.sum(after > before)
        same = np.sum(after == before)
        degraded = np.sum(after < before)

        print(f"\nBatch-level comparison:")
        print(f"  Improved (child > parent): {improved} ({100*improved/len(batch_df):.1f}%)")
        print(f"  Same: {same} ({100*same/len(batch_df):.1f}%)")
        print(f"  Degraded (child < parent): {degraded} ({100*degraded/len(batch_df):.1f}%)")

        print(f"\n  Mean batch score before: {np.mean(before):.4f}")
        print(f"  Mean batch score after: {np.mean(after):.4f}")
        print(f"  Mean improvement: {np.mean(after - before):.4f}")


def main():
    data = fetch_all_data()

    analyze_score_progression(data)
    analyze_minibatch_variation(data)
    analyze_parent_child_comparison(data)
    analyze_sample_level_similarity(data)
    analyze_batch_scores(data)


if __name__ == "__main__":
    main()
