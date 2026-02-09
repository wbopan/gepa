"""Analyze wandb runs for adaboost vs adaboost-avg comparison."""

import wandb
import pandas as pd
import numpy as np
from collections import Counter

# Initialize wandb API
api = wandb.Api()

# Run configurations
RUNS = {
    "adaboost": "nfadkixt",
    "adaboost-avg": "pfng4pwq"
}
PROJECT = "bmpixel/gepa-boost"

def fetch_run_data(run_name: str, run_id: str):
    """Fetch history and artifacts for a run."""
    print(f"\n{'='*60}")
    print(f"Fetching data for run: {run_name} ({run_id})")
    print('='*60)
    
    run = api.run(f"{PROJECT}/{run_id}")
    
    # Get artifacts - get the latest version
    artifacts = list(run.logged_artifacts())
    candidate_prompts_artifact = None
    for art in artifacts:
        if "candidate_prompts" in art.name:
            candidate_prompts_artifact = art  # Will get the last (latest) one
    
    if candidate_prompts_artifact:
        print(f"Using artifact: {candidate_prompts_artifact.name}")
    
    # Download and load the table
    table_data = None
    if candidate_prompts_artifact:
        table = candidate_prompts_artifact.get("candidate_prompts")
        if table:
            table_data = pd.DataFrame(data=table.data, columns=table.columns)
            print(f"Table rows: {len(table_data)}")
            print(f"Table columns: {list(table_data.columns)}")
    
    return {
        "table": table_data,
        "run": run
    }

def analyze_score_progression(data: dict, run_name: str):
    """Analyze score progression over iterations."""
    print(f"\n--- Score Progression for {run_name} ---")
    table = data["table"]
    
    if table is not None and "valset_aggregate_score" in table.columns:
        table_scores = table["valset_aggregate_score"].dropna()
        print(f"  Total candidates: {len(table_scores)}")
        print(f"  Min score: {table_scores.min():.4f}")
        print(f"  Max score: {table_scores.max():.4f}")
        print(f"  Mean score: {table_scores.mean():.4f}")
        print(f"  Std score: {table_scores.std():.4f}")
        
        # Show score progression by candidate index
        print(f"\n  Score by candidate index:")
        for idx, row in table.iterrows():
            score = row.get("valset_aggregate_score")
            parent = row.get("parent_idx")
            cand_idx = row.get("candidate_idx")
            if score is not None:
                parent_str = f"parent={int(parent)}" if parent is not None and not (isinstance(parent, float) and np.isnan(parent)) else "seed"
                print(f"    Candidate {cand_idx}: score={score:.4f}, {parent_str}")

def analyze_parent_selection(data: dict, run_name: str):
    """Analyze parent selection diversity."""
    print(f"\n--- Parent Selection Analysis for {run_name} ---")
    table = data["table"]
    
    if table is not None and "parent_idx" in table.columns:
        parent_idx = table["parent_idx"].dropna()
        parent_counts = Counter()
        for p in parent_idx:
            if isinstance(p, (list, np.ndarray)):
                for pp in p:
                    if pp is not None and not (isinstance(pp, float) and np.isnan(pp)):
                        parent_counts[int(pp)] += 1
            elif p is not None and not (isinstance(p, float) and np.isnan(p)):
                parent_counts[int(p)] += 1
        
        print(f"  Total parent references: {sum(parent_counts.values())}")
        print(f"  Unique parents: {len(parent_counts)}")
        print(f"  Most common parents: {parent_counts.most_common(10)}")
        
        if parent_counts:
            values = np.array(list(parent_counts.values()))
            total = values.sum()
            probs = values / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(parent_counts)) if len(parent_counts) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            print(f"  Selection entropy: {entropy:.4f}")
            print(f"  Normalized entropy: {normalized_entropy:.4f} (1=uniform, 0=concentrated)")
            
            top_parent_count = parent_counts.most_common(1)[0][1]
            print(f"  Top parent selection rate: {100*top_parent_count/total:.1f}%")
    
    return parent_counts if table is not None and "parent_idx" in table.columns else None

def analyze_parent_child_scores(data: dict, run_name: str):
    """Analyze parent-child score relationships."""
    print(f"\n--- Parent-Child Score Relationships for {run_name} ---")
    table = data["table"]
    
    if table is None:
        print("  No table data available")
        return
    
    score_col = "valset_aggregate_score"
    if score_col not in table.columns:
        print(f"  Score column {score_col} not found")
        return
    
    improvements = []
    parent_child_pairs = []
    
    for idx, row in table.iterrows():
        parent = row.get("parent_idx")
        child_score = row.get(score_col)
        child_idx = row.get("candidate_idx")
        
        if parent is not None and child_score is not None:
            try:
                if isinstance(parent, (list, np.ndarray)):
                    parent = parent[0] if len(parent) > 0 else None
                if parent is not None and not (isinstance(parent, float) and np.isnan(parent)):
                    parent_idx = int(parent)
                    parent_row = table[table["candidate_idx"] == parent_idx]
                    if len(parent_row) > 0:
                        parent_score = parent_row.iloc[0].get(score_col)
                        if parent_score is not None and not np.isnan(parent_score) and not np.isnan(child_score):
                            improvement = child_score - parent_score
                            improvements.append(improvement)
                            parent_child_pairs.append({
                                "parent_idx": parent_idx,
                                "parent_score": parent_score,
                                "child_idx": child_idx,
                                "child_score": child_score,
                                "improvement": improvement
                            })
            except (ValueError, TypeError, IndexError):
                continue
    
    if improvements:
        print(f"  Score improvements (child - parent):")
        print(f"    Count: {len(improvements)}")
        print(f"    Mean improvement: {np.mean(improvements):.4f}")
        print(f"    Median improvement: {np.median(improvements):.4f}")
        print(f"    Std improvement: {np.std(improvements):.4f}")
        print(f"    Min improvement: {np.min(improvements):.4f}")
        print(f"    Max improvement: {np.max(improvements):.4f}")
        print(f"    Positive improvements: {sum(1 for i in improvements if i > 0)} ({100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%)")
        print(f"    Zero improvements: {sum(1 for i in improvements if i == 0)} ({100*sum(1 for i in improvements if i == 0)/len(improvements):.1f}%)")
        print(f"    Negative improvements: {sum(1 for i in improvements if i < 0)} ({100*sum(1 for i in improvements if i < 0)/len(improvements):.1f}%)")
        
        print(f"\n  All parent-child relationships:")
        for pc in sorted(parent_child_pairs, key=lambda x: x["child_idx"]):
            sign = "+" if pc["improvement"] > 0 else ""
            print(f"    Parent {pc['parent_idx']} ({pc['parent_score']:.4f}) -> Child {pc['child_idx']} ({pc['child_score']:.4f}): {sign}{pc['improvement']:.4f}")

def analyze_family_trees(data: dict, run_name: str):
    """Analyze family tree structure."""
    print(f"\n--- Family Tree Analysis for {run_name} ---")
    table = data["table"]
    
    if table is None:
        print("  No table data available")
        return
    
    score_col = "valset_aggregate_score"
    
    children_by_parent = {}
    for idx, row in table.iterrows():
        parent = row.get("parent_idx")
        child_idx = row.get("candidate_idx")
        child_score = row.get(score_col)
        
        if parent is not None and not (isinstance(parent, float) and np.isnan(parent)):
            parent_idx = int(parent)
            if parent_idx not in children_by_parent:
                children_by_parent[parent_idx] = []
            children_by_parent[parent_idx].append({
                "child_idx": child_idx,
                "child_score": child_score
            })
    
    print(f"  Parents with children: {len(children_by_parent)}")
    for parent_idx in sorted(children_by_parent.keys()):
        parent_row = table[table["candidate_idx"] == parent_idx]
        if len(parent_row) > 0:
            parent_score = parent_row.iloc[0].get(score_col)
            children = children_by_parent[parent_idx]
            child_scores = [c["child_score"] for c in children if c["child_score"] is not None]
            
            if child_scores and parent_score is not None:
                avg_child_score = np.mean(child_scores)
                best_child_score = np.max(child_scores)
                better_children = sum(1 for s in child_scores if s > parent_score)
                
                print(f"    Parent {parent_idx} (score={parent_score:.4f}): {len(children)} children, "
                      f"avg={avg_child_score:.4f}, best={best_child_score:.4f}, "
                      f"better={better_children}/{len(children)}")

def compare_runs(all_data: dict):
    """Compare statistics between runs."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    
    comparison_stats = {}
    
    for run_name, data in all_data.items():
        table = data["table"]
        stats = {}
        
        if table is not None and "valset_aggregate_score" in table.columns:
            scores = table["valset_aggregate_score"].dropna()
            stats["best_score"] = scores.max()
            stats["mean_score"] = scores.mean()
            stats["total_candidates"] = len(table)
            
            if "parent_idx" in table.columns:
                parent_counts = Counter()
                for p in table["parent_idx"].dropna():
                    if not (isinstance(p, float) and np.isnan(p)):
                        parent_counts[int(p)] += 1
                
                stats["unique_parents"] = len(parent_counts)
                if parent_counts:
                    values = np.array(list(parent_counts.values()))
                    total = values.sum()
                    probs = values / total
                    stats["selection_entropy"] = -np.sum(probs * np.log2(probs + 1e-10))
                    stats["top_parent_rate"] = parent_counts.most_common(1)[0][1] / total
        
        comparison_stats[run_name] = stats
    
    print("\nMetric Comparison:")
    print("-" * 50)
    
    metrics = ["total_candidates", "best_score", "mean_score", "unique_parents", "selection_entropy", "top_parent_rate"]
    metric_labels = {
        "total_candidates": "Total Candidates",
        "best_score": "Best Score",
        "mean_score": "Mean Score", 
        "unique_parents": "Unique Parents Used",
        "selection_entropy": "Selection Entropy (higher=more diverse)",
        "top_parent_rate": "Top Parent Selection Rate"
    }
    
    for metric in metrics:
        print(f"\n{metric_labels.get(metric, metric)}:")
        for run_name in all_data.keys():
            val = comparison_stats[run_name].get(metric)
            if val is not None:
                if isinstance(val, float):
                    print(f"  {run_name}: {val:.4f}")
                else:
                    print(f"  {run_name}: {val}")

def main():
    all_data = {}
    
    for run_name, run_id in RUNS.items():
        all_data[run_name] = fetch_run_data(run_name, run_id)
    
    for run_name, data in all_data.items():
        analyze_score_progression(data, run_name)
        analyze_parent_selection(data, run_name)
        analyze_parent_child_scores(data, run_name)
        analyze_family_trees(data, run_name)
    
    compare_runs(all_data)

if __name__ == "__main__":
    main()
