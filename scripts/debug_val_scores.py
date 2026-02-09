#!/usr/bin/env python3
"""Debug: understand validation score structure in wandb."""

import wandb
from collections import defaultdict

api = wandb.Api()
entity = "bmpixel"
project = "gepa-boost"

# Get one run to understand structure
run = api.run(f"{entity}/{project}/znn618br")
print(f"Run: {run.displayName}")

# Fetch history and look at rows with val/new_score
history = run.scan_history()

iteration_data = defaultdict(dict)
for row in history:
    iteration = row.get("gepa_iteration")
    if iteration is None:
        continue

    # Collect all relevant fields
    for key in ["val/new_score", "candidate/new_idx", "candidate/selected_idx",
                "train/batch_score_before", "train/batch_score_after"]:
        val = row.get(key)
        if val is not None:
            iteration_data[iteration][key] = val

# Show first 20 iterations
print("\nFirst 20 iterations with data:")
print("-" * 80)
for i in sorted(iteration_data.keys())[:20]:
    data = iteration_data[i]
    print(f"Iteration {i}: {data}")

# Count how many iterations have val/new_score
val_score_iters = [i for i, d in iteration_data.items() if "val/new_score" in d]
print(f"\n{len(val_score_iters)} iterations have val/new_score")

# Show iterations where val/new_score is logged
print("\nIterations with val/new_score:")
print("-" * 80)
for i in sorted(val_score_iters)[:10]:
    data = iteration_data[i]
    print(f"Iteration {i}:")
    print(f"  val/new_score: {data.get('val/new_score')}")
    print(f"  candidate/new_idx: {data.get('candidate/new_idx')}")
    print(f"  candidate/selected_idx: {data.get('candidate/selected_idx')}")
