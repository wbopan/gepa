"""Fetch valset_outputs from wandb runs for Pareto frontier analysis."""

import json
from pathlib import Path

import pandas as pd
import wandb

# Configuration
ENTITY = "bmpixel"
PROJECT = "gepa-boost"
RUNS = {
    "adaboost": "nfadkixt",
    "bayesian": "k0jcelpb",
    "baseline": "ipnfzdjs",
}


def fetch_valset_outputs(run_id: str, run_name: str) -> pd.DataFrame | None:
    """Fetch valset_outputs table from a wandb run."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    print(f"Fetching valset_outputs for {run_name} ({run_id})...")

    # Find the valset_outputs artifact (look for the latest one)
    valset_artifact = None
    for artifact in run.logged_artifacts():
        if "valset_outputs" in artifact.name:
            valset_artifact = artifact

    if valset_artifact is None:
        print(f"  No valset_outputs artifact found for {run_name}")
        return None

    print(f"  Found artifact: {valset_artifact.name}")

    # Download and load the table
    try:
        table = valset_artifact.get("valset_outputs")
        if table:
            df = pd.DataFrame(data=table.data, columns=table.columns)
            print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
            return df
    except Exception as e:
        print(f"  Error loading table: {e}")

        # Alternative: download the artifact directory
        try:
            artifact_dir = valset_artifact.download()
            table_file = Path(artifact_dir) / "valset_outputs.table.json"
            if table_file.exists():
                with open(table_file) as f:
                    table_data = json.load(f)
                columns = table_data["columns"]
                df = pd.DataFrame(table_data["data"], columns=columns)
                print(f"  Loaded {len(df)} rows from file, columns: {list(df.columns)}")
                return df
        except Exception as e2:
            print(f"  Error loading from file: {e2}")

    return None


def main():
    output_dir = Path("/Users/panwenbo/Repos/gepa/analysis_output")
    output_dir.mkdir(exist_ok=True)

    all_data = {}

    for run_name, run_id in RUNS.items():
        df = fetch_valset_outputs(run_id, run_name)
        if df is not None:
            all_data[run_name] = df
            # Save to CSV
            csv_path = output_dir / f"valset_outputs_{run_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved to {csv_path}")

            # Save to JSON for raw data
            json_path = output_dir / f"valset_outputs_{run_name}.json"
            df.to_json(json_path, orient="records", indent=2)
            print(f"  Saved to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for run_name, df in all_data.items():
        print(f"\n{run_name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        if "candidate_idx" in df.columns:
            print(f"  Unique candidates: {df['candidate_idx'].nunique()}")
        if "sample_idx" in df.columns or "sample_id" in df.columns:
            sample_col = "sample_idx" if "sample_idx" in df.columns else "sample_id"
            print(f"  Unique samples: {df[sample_col].nunique()}")

    return all_data


if __name__ == "__main__":
    main()
