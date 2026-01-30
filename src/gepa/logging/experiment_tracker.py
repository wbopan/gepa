# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import os
from typing import Any

import weave


class ExperimentTracker:
    """Experiment tracking using wandb weave."""

    def __init__(
        self,
        use_weave: bool = False,
        weave_project_name: str | None = None,
    ):
        self.use_weave = use_weave
        self.weave_project_name = weave_project_name or "gepa-boost"
        self._prompt_refs: dict[int, str] = {}  # candidate_idx -> AcceptedPrompt ref

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def start_run(self):
        """Initialize weave and start wandb run."""
        if self.use_weave:
            import wandb

            # Suppress weave trace URL printing by default
            os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")

            wandb.login()
            wandb.init(project=self.weave_project_name)

            # Configure wandb to group metrics by gepa_iteration field
            wandb.define_metric("gepa_iteration")
            wandb.define_metric("*", step_metric="gepa_iteration")

            # Initialize weave with tracing enabled
            weave.init(project_name=self.weave_project_name)
        else:
            # Initialize weave with tracing disabled
            weave.init(project_name=self.weave_project_name, settings={"disabled": True})

    def log_metrics(self, metrics: dict[str, Any], iteration: int | None = None):
        """Log time-series metrics to wandb.

        Args:
            metrics: Dictionary of metric name to value.
            iteration: The GEPA iteration number. When provided, all metrics logged
                      with the same iteration value will be grouped together in wandb.
        """
        if self.use_weave:
            try:
                import wandb

                if iteration is not None:
                    metrics = {**metrics, "gepa_iteration": iteration}
                wandb.log(metrics)
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

    def publish_prompt(
        self,
        content: dict[str, str],
        iteration: int,
        parent_ref: str | None = None,
        minibatch_score_before: float = 0.0,
        minibatch_score_after: float = 0.0,
        accepted: bool = False,
        candidate_idx: int | None = None,
        valset_score: float | None = None,
    ) -> str | None:
        """Publish a GEPAPrompt and return its ref URI."""
        if not self.use_weave:
            return None
        try:
            from gepa.logging.prompt_tracker import GEPAPrompt

            prompt = GEPAPrompt(
                content=content,
                iteration=iteration,
                parent_ref=parent_ref,
                minibatch_score_before=minibatch_score_before,
                minibatch_score_after=minibatch_score_after,
                accepted=accepted,
                candidate_idx=candidate_idx,
                valset_score=valset_score,
            )
            ref = weave.publish(prompt)
            ref_uri = str(ref.uri())

            if accepted and candidate_idx is not None:
                self._prompt_refs[candidate_idx] = ref_uri

            return ref_uri
        except Exception as e:
            print(f"Warning: Failed to publish prompt: {e}")
            return None

    def get_prompt_ref(self, candidate_idx: int) -> str | None:
        """Get the AcceptedPrompt ref for a candidate index."""
        return self._prompt_refs.get(candidate_idx)

    def log_final_results(
        self,
        best_candidate: dict[str, str],
        best_candidate_idx: int,
        best_score: float,
        total_candidates: int,
        total_metric_calls: int,
    ) -> None:
        """Save final optimization results to wandb summary."""
        self.log_summary(
            {
                "best_candidate_idx": best_candidate_idx,
                "best_score": best_score,
                "total_candidates": total_candidates,
                "total_metric_calls": total_metric_calls,
                "best_candidate": best_candidate,
            }
        )


def create_experiment_tracker(
    use_weave: bool = False,
    weave_project_name: str | None = None,
) -> ExperimentTracker:
    return ExperimentTracker(
        use_weave=use_weave,
        weave_project_name=weave_project_name,
    )
