# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any


class ExperimentTracker:
    """Experiment tracking using wandb weave."""

    def __init__(
        self,
        use_weave: bool = False,
        weave_project_name: str | None = None,
    ):
        self.use_weave = use_weave
        self.weave_project_name = weave_project_name or "gepa-optimization"

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def start_run(self):
        """Initialize weave and start wandb run."""
        if self.use_weave:
            try:
                import os

                # Suppress weave trace URL printing by default
                os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")

                import weave  # noqa: F401 weave auto-patches litellm when imported with wandb

                import wandb

                wandb.login()
                wandb.init(project=self.weave_project_name)
            except ImportError:
                raise ImportError("weave is not installed. Install with: pip install 'weave[litellm]'")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log time-series metrics to wandb."""
        if self.use_weave:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")

    def log_summary(self, metrics: dict[str, Any]):
        """Log one-time summary metrics to wandb."""
        if self.use_weave:
            try:
                import wandb

                for key, value in metrics.items():
                    wandb.summary[key] = value
            except Exception as e:
                print(f"Warning: Failed to log summary: {e}")

    def end_run(self):
        """End the wandb run."""
        if self.use_weave:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to end run: {e}")

    def is_active(self) -> bool:
        """Check if tracking is active."""
        if self.use_weave:
            try:
                import wandb

                return wandb.run is not None
            except Exception:
                pass
        return False

    def log_prompt_artifact(
        self,
        prompt: dict[str, str],
        candidate_idx: int,
        iteration: int,
        is_best: bool = False,
        parent_idx: int | None = None,
        valset_score: float | None = None,
    ) -> None:
        """Save candidate prompt as wandb artifact."""
        if not self.use_weave:
            return
        try:
            import wandb

            # Create text content
            lines = [f"# Candidate {candidate_idx} - Iteration {iteration}"]
            if parent_idx is not None:
                lines.append(f"# Parent: Candidate {parent_idx}")
            if valset_score is not None:
                lines.append(f"# Valset Score: {valset_score:.4f}")
            lines.append("")
            for component_name, component_text in prompt.items():
                lines.append(f"## {component_name}")
                lines.append(component_text)
                lines.append("")

            # Log as table for easy viewing
            table = wandb.Table(columns=["component", "text"])
            for component_name, component_text in prompt.items():
                table.add_data(component_name, component_text)
            wandb.log({f"prompts/candidate_{candidate_idx:03d}": table})

            # Update summary if best
            if is_best:
                wandb.summary["best_prompt"] = prompt
                wandb.summary["best_candidate_idx"] = candidate_idx
        except Exception as e:
            print(f"Warning: Failed to log prompt artifact: {e}")

    def log_score_distribution(
        self,
        scores_by_val_id: dict,
        candidate_idx: int,
        iteration: int,
        objective_scores: dict | None = None,
    ) -> None:
        """Save per-example scores as wandb table."""
        if not self.use_weave:
            return
        try:
            import wandb

            table = wandb.Table(columns=["val_id", "score"])
            for val_id, score in scores_by_val_id.items():
                table.add_data(str(val_id), score)
            wandb.log({f"scores/candidate_{candidate_idx:03d}": table})

            if objective_scores:
                wandb.log({f"objective_scores/candidate_{candidate_idx}": objective_scores})
        except Exception as e:
            print(f"Warning: Failed to log score distribution: {e}")

    def log_final_results(
        self,
        best_candidate: dict[str, str],
        best_candidate_idx: int,
        best_score: float,
        total_candidates: int,
        total_metric_calls: int,
    ) -> None:
        """Save final optimization results to wandb summary."""
        if not self.use_weave:
            return
        try:
            import wandb

            wandb.summary["best_candidate_idx"] = best_candidate_idx
            wandb.summary["best_score"] = best_score
            wandb.summary["total_candidates"] = total_candidates
            wandb.summary["total_metric_calls"] = total_metric_calls
            wandb.summary["best_candidate"] = best_candidate
        except Exception as e:
            print(f"Warning: Failed to log final results: {e}")


def create_experiment_tracker(
    use_weave: bool = False,
    weave_project_name: str | None = None,
) -> ExperimentTracker:
    return ExperimentTracker(
        use_weave=use_weave,
        weave_project_name=weave_project_name,
    )
