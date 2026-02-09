"""Read and display the best parent prompts for qualitative analysis."""

import json
from pathlib import Path

import wandb


def main():
    entity = "bmpixel"
    project = "gepa-boost"

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # Collect all candidates from all runs
    all_candidates = {}

    for run in runs:
        for artifact in run.logged_artifacts():
            if "candidate_prompts" in artifact.name:
                artifact_dir = artifact.download()
                table_file = Path(artifact_dir) / "candidate_prompts.table.json"
                if table_file.exists():
                    with open(table_file) as f:
                        table_data = json.load(f)
                    columns = table_data["columns"]
                    rows = [dict(zip(columns, row)) for row in table_data["data"]]

                    # Keep the latest (largest) version
                    for row in rows:
                        idx = row["candidate_idx"]
                        if idx not in all_candidates or len(rows) > len(all_candidates):
                            all_candidates[idx] = row

    # Best parents identified: [3, 16, 7, 5, 15]
    best_parent_ids = [3, 16, 7, 5, 15, 0, 29]  # Include seed (0) and high improvement rate (29)

    print("=" * 80)
    print("DETAILED ANALYSIS OF BEST PARENT PROMPTS")
    print("=" * 80)

    for parent_idx in best_parent_ids:
        if parent_idx not in all_candidates:
            continue

        candidate = all_candidates[parent_idx]
        try:
            content = json.loads(candidate["candidate_content"])
            prompt = content.get("system_prompt", "")
        except (json.JSONDecodeError, TypeError):
            continue

        print(f"\n{'='*80}")
        print(f"PARENT #{parent_idx}")
        print(f"{'='*80}")
        print(f"Score: {candidate['valset_aggregate_score']:.3f}")
        print(f"Length: {len(prompt)} chars")
        print(f"\n--- FULL PROMPT ---\n")
        print(prompt)
        print(f"\n--- END ---\n")


if __name__ == "__main__":
    main()
