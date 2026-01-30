# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any


class ExperimentTracker:
    """
    Unified experiment tracking that supports both wandb and mlflow.
    """

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always end the run."""
        self.end_run()
        return False  # Don't suppress exceptions

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs or {}
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        self._created_mlflow_run = False

    def initialize(self):
        """Initialize the logging backends."""
        if self.use_wandb:
            self._initialize_wandb()
        if self.use_mlflow:
            self._initialize_mlflow()

    def _initialize_wandb(self):
        """Initialize wandb."""
        try:
            import wandb  # type: ignore

            if self.wandb_api_key:
                wandb.login(key=self.wandb_api_key, verify=True)
            else:
                wandb.login()
        except ImportError:
            raise ImportError("wandb is not installed. Please install it or set backend='mlflow' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error logging into wandb: {e}")

    def _initialize_mlflow(self):
        """Initialize mlflow."""
        try:
            import mlflow  # type: ignore

            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)
        except ImportError:
            raise ImportError("mlflow is not installed. Please install it or set backend='wandb' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error setting up mlflow: {e}")

    def start_run(self):
        """Start a new run."""
        if self.use_wandb:
            import wandb  # type: ignore

            wandb.init(**self.wandb_init_kwargs)
        if self.use_mlflow:
            import mlflow  # type: ignore

            # Only start a new run if there's no active run
            if mlflow.active_run() is None:
                mlflow.start_run()
                self._created_mlflow_run = True
            else:
                self._created_mlflow_run = False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to the active backends."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                # MLflow only accepts numeric metrics, filter out non-numeric values
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to mlflow: {e}")

    def end_run(self):
        """End the current run."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to end wandb run: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                if self._created_mlflow_run and mlflow.active_run() is not None:
                    mlflow.end_run()
                    self._created_mlflow_run = False
            except Exception as e:
                print(f"Warning: Failed to end mlflow run: {e}")

    def is_active(self) -> bool:
        """Check if any backend has an active run."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    return True
            except Exception:
                pass

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                if mlflow.active_run() is not None:
                    return True
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
        """Save candidate prompt as a text artifact."""
        if not self.use_mlflow:
            return
        try:
            import mlflow

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

            filename = "prompts/best_prompt.txt" if is_best else f"prompts/candidate_{candidate_idx:03d}.txt"
            mlflow.log_text("\n".join(lines), filename)
        except Exception as e:
            print(f"Warning: Failed to log prompt artifact: {e}")

    def log_score_distribution(
        self,
        scores_by_val_id: dict,
        candidate_idx: int,
        iteration: int,
        objective_scores: dict | None = None,
    ) -> None:
        """Save per-example raw scores as a JSON artifact."""
        if not self.use_mlflow:
            return
        try:
            import mlflow

            data = {
                "candidate_idx": candidate_idx,
                "iteration": iteration,
                "num_examples": len(scores_by_val_id),
                "scores_by_val_id": {str(k): v for k, v in scores_by_val_id.items()},
            }
            if objective_scores:
                data["objective_scores"] = dict(objective_scores)

            mlflow.log_dict(data, f"scores/candidate_{candidate_idx:03d}.json")
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
        """Save final optimization results."""
        if not self.use_mlflow:
            return
        try:
            import mlflow

            # Save best prompt
            self.log_prompt_artifact(
                prompt=best_candidate,
                candidate_idx=best_candidate_idx,
                iteration=-1,
                is_best=True,
                valset_score=best_score,
            )

            # Save summary
            summary = {
                "best_candidate_idx": best_candidate_idx,
                "best_score": best_score,
                "total_candidates": total_candidates,
                "total_metric_calls": total_metric_calls,
            }
            mlflow.log_dict(summary, "final_summary.json")
        except Exception as e:
            print(f"Warning: Failed to log final results: {e}")


def create_experiment_tracker(
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> ExperimentTracker:
    """
    Create an experiment tracker based on the specified backends.

    Args:
        use_wandb: Whether to use wandb
        use_mlflow: Whether to use mlflow
        wandb_api_key: API key for wandb
        wandb_init_kwargs: Additional kwargs for wandb.init()
        mlflow_tracking_uri: Tracking URI for mlflow
        mlflow_experiment_name: Experiment name for mlflow

    Returns:
        ExperimentTracker instance

    Note:
        Both wandb and mlflow can be used simultaneously if desired.
    """
    return ExperimentTracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
