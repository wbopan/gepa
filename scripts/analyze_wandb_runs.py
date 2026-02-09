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
    
    # Get full history
    history = run.history(keys=["val/new_score", "candidate/selected_idx"], pandas=True)
    print(f"History rows: {len(history)}")
    
    # Get artifacts
    artifacts = run.logged_artifacts()
    candidate_prompts_artifact = None
    for art in artifacts:
        if "candidate_prompts" in art.name:
            candidate_prompts_artifact = art
            print(f"Found artifact: {art.name}")
    
    # Download and load the table
    table_data = None
    if candidate_prompts_artifact:
        table = candidate_prompts_artifact.get("candidate_prompts")
        if table:
            table_data = pd.DataFrame(data=table.data, columns=table.columns)
            print(f"Table rows: {len(table_data)}")
            print(f"Table columns: {list(table_data.columns)}")
    
    return {
        "history": history,
        "table": table_data,
        "run": run
    }

def analyze_score_progression(data: dict, run_name: str):
    """Analyze score progression over iterations."""
    print(f"\n--- Score Progression for {run_name} ---")
    history = data["history"]
    
    if "val/new_score" in history.columns:
        scores = history["val/new_score"].dropna()
        print(f"  Total score entries: {len(scores)}")
        print(f"  Min score: {scores.min():.4f}")
        print(f"  Max score: {scores.max():.4f}")
        print(f"  Mean score: {scores.mean():.4f}")
        print(f"  Std score: {scores.std():.4f}")
        
        # Score progression by chunks (iterations)
        chunk_size = max(1, len(scores) // 10)
        for i in range(0, len(scores), chunk_size):
            chunk = scores.iloc[i:i+chunk_size]
            print(f"  Iterations {i}-{i+len(chunk)-1}: mean={chunk.mean():.4f}, max={chunk.max():.4f}")
    
    return scores if "val/new_score" in history.columns else None

def analyze_parent_selection(data: dict, run_name: str):
    """Analyze parent selection diversity."""
    print(f"\n--- Parent Selection Analysis for {run_name} ---")
    history = data["history"]
    table = data["table"]
    
    # From history - selected_idx
    if "candidate/selected_idx" in history.columns:
        selected = history["candidate/selected_idx"].dropna().astype(int)
        counts = Counter(selected)
        print(f"  Selected indices from history:")
        print(f"    Total selections: {len(selected)}")
        print(f"    Unique indices selected: {len(counts)}")
        print(f"    Most common selections: {counts.most_common(10)}")
        
        # Distribution stats
        selection_counts = list(counts.values())
        print(f"    Selection count range: {min(selection_counts)} - {max(selection_counts)}")
        print(f"    Mean selections per candidate: {np.mean(selection_counts):.2f}")
        print(f"    Std selections: {np.std(selection_counts):.2f}")
    
    # From table - parent_idx relationships
    if table is not None and "parent_idx" in table.columns:
        parent_idx = table["parent_idx"].dropna()
        # Handle potential list/array values
        parent_counts = Counter()
        for p in parent_idx:
            if isinstance(p, (list, np.ndarray)):
                for pp in p:
                    if pp is not None and not (isinstance(pp, float) and np.isnan(pp)):
                        parent_counts[int(pp)] += 1
            elif p is not None and not (isinstance(p, float) and np.isnan(p)):
                parent_counts[int(p)] += 1
        
        print(f"\n  Parent indices from table:")
        print(f"    Total parent references: {sum(parent_counts.values())}")
        print(f"    Unique parents: {len(parent_counts)}")
        print(f"    Most common parents: {parent_counts.most_common(10)}")
    
    return counts if "candidate/selected_idx" in history.columns else None

def analyze_parent_child_scores(data: dict, run_name: str):
    """Analyze parent-child score relationships."""
    print(f"\n--- Parent-Child Score Relationships for {run_name} ---")
    table = data["table"]
    
    if table is None:
        print("  No table data available")
        return
    
    # Check available columns
    score_cols = [c for c in table.columns if "score" in c.lower()]
    print(f"  Score columns: {score_cols}")
    
    if "parent_idx" in table.columns and len(score_cols) > 0:
        score_col = score_cols[0]  # Use first score column
        
        improvements = []
        for idx, row in table.iterrows():
            parent = row.get("parent_idx")
            child_score = row.get(score_col)
            
            if parent is not None and child_score is not None:
                try:
                    if isinstance(parent, (list, np.ndarray)):
                        parent = parent[0] if len(parent) > 0 else None
                    if parent is not None and not (isinstance(parent, float) and np.isnan(parent)):
                        parent_idx = int(parent)
                        if parent_idx < len(table):
                            parent_score = table.iloc[parent_idx].get(score_col)
                            if parent_score is not None and not np.isnan(parent_score) and not np.isnan(child_score):
                                improvement = child_score - parent_score
                                improvements.append(improvement)
                except (ValueError, TypeError, IndexError):
                    continue
        
        if improvements:
            print(f"  Score improvements (child - parent):")
            print(f"    Count: {len(improvements)}")
            print(f"    Mean improvement: {np.mean(improvements):.4f}")
            print(f"    Median improvement: {np.median(improvements):.4f}")
            print(f"    Positive improvements: {sum(1 for i in improvements if i > 0)} ({100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%)")
            print(f"    Negative improvements: {sum(1 for i in improvements if i < 0)} ({100*sum(1 for i in improvements if i < 0)/len(improvements):.1f}%)")

def compare_runs(all_data: dict):
    """Compare statistics between runs."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    
    for run_name, data in all_data.items():
        history = data["history"]
        table = data["table"]
        
        print(f"\n{run_name}:")
        if "val/new_score" in history.columns:
            scores = history["val/new_score"].dropna()
            print(f"  Final mean score: {scores.tail(20).mean():.4f}")
            print(f"  Best score: {scores.max():.4f}")
        
        if "candidate/selected_idx" in history.columns:
            selected = history["candidate/selected_idx"].dropna().astype(int)
            counts = Counter(selected)
            # Gini coefficient for selection diversity
            values = np.array(list(counts.values()))
            if len(values) > 1:
                sorted_vals = np.sort(values)
                n = len(sorted_vals)
                cumsum = np.cumsum(sorted_vals)
                gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals))) / (n * np.sum(sorted_vals)) - (n + 1) / n
                print(f"  Selection Gini coefficient: {gini:.4f} (0=equal, 1=concentrated)")
            print(f"  Selection entropy: {-sum(c/sum(counts.values()) * np.log2(c/sum(counts.values())) for c in counts.values()):.4f}")

def main():
    all_data = {}
    
    # Fetch data for both runs
    for run_name, run_id in RUNS.items():
        all_data[run_name] = fetch_run_data(run_name, run_id)
    
    # Analyze each run
    for run_name, data in all_data.items():
        analyze_score_progression(data, run_name)
        analyze_parent_selection(data, run_name)
        analyze_parent_child_scores(data, run_name)
    
    # Compare runs
    compare_runs(all_data)

if __name__ == "__main__":
    main()
