# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.logging.weave_tracing import configure_weave_tracing


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
            try:
                import os

                # Suppress weave trace URL printing by default
                os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")

                import weave  # noqa: F401 weave auto-patches litellm when imported with wandb

                import wandb

                wandb.login()
                wandb.init(project=self.weave_project_name)

                # Configure weave tracing for hierarchical call organization
                configure_weave_tracing(enabled=True, client=weave)
            except ImportError:
                raise ImportError("weave is not installed. Install with: pip install 'weave[litellm]'")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, commit: bool = True):
        """Log time-series metrics to wandb.

        Args:
            metrics: Dictionary of metric name to value.
            step: The step number for this log entry.
            commit: If True (default), finalize this step. If False, allow additional
                   metrics to be logged at the same step before committing.
                   Use commit=False when logging multiple metrics at the same step,
                   then commit=True on the final log call for that step.
        """
        if self.use_weave:
            try:
                import wandb

                wandb.log(metrics, step=step, commit=commit)
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

                # Disable weave tracing
                configure_weave_tracing(enabled=False, client=None)

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

    def publish_proposed_prompt(
        self,
        content: dict[str, str],
        iteration: int,
        parent_ref: str | None,
        minibatch_score_before: float,
        minibatch_score_after: float,
        accepted: bool,
    ) -> str | None:
        """Publish a ProposedPrompt and return its ref URI."""
        if not self.use_weave:
            return None
        try:
            import weave

            from gepa.logging.prompt_tracker import ProposedPrompt

            prompt = ProposedPrompt(
                content=content,
                parent_ref=parent_ref,
                iteration=iteration,
                minibatch_score_before=minibatch_score_before,
                minibatch_score_after=minibatch_score_after,
                accepted=accepted,
            )
            ref = weave.publish(prompt)
            return str(ref.uri())
        except Exception as e:
            print(f"Warning: Failed to publish proposed prompt: {e}")
            return None

    def publish_accepted_prompt(
        self,
        proposed_ref: str,
        candidate_idx: int,
        valset_score: float,
    ) -> str | None:
        """Publish an AcceptedPrompt and return its ref URI."""
        if not self.use_weave:
            return None
        try:
            import weave

            from gepa.logging.prompt_tracker import AcceptedPrompt

            prompt = AcceptedPrompt(
                proposed_ref=proposed_ref,
                candidate_idx=candidate_idx,
                valset_score=valset_score,
            )
            ref = weave.publish(prompt)
            ref_uri = str(ref.uri())
            self._prompt_refs[candidate_idx] = ref_uri
            return ref_uri
        except Exception as e:
            print(f"Warning: Failed to publish accepted prompt: {e}")
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
