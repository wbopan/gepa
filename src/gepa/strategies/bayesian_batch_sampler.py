# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass, field
from typing import Sequence

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.logging import get_logger
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.residual_weighted_sampler import ResidualWeightedSampler

logger = get_logger()


def bayesian_frontier_score(successes: float, failures: float) -> float:
    """Compute Bayesian variance-based sampling priority score.

    Uses 4 * Var(Bernoulli) with Beta(1,1) prior (Laplace smoothing).
    This identifies "frontier" samples where candidates show maximum disagreement.

    Formula: 4 * (s + 1) * (f + 1) / (s + f + 2)^2

    Args:
        successes: Number of times the sample was correctly solved.
        failures: Number of times the sample was incorrectly solved.

    Returns:
        Priority score in (0, 1]. 1.0 = highest priority (cold start or perfect frontier).

    Design intuition:
        - Cold start (0, 0): 1.0 - unexplored samples are potential frontiers
        - Perfect frontier (s = f): 1.0 - maximum disagreement, strongest learning signal
        - One-sided (s >> f or f >> s): low score - sample has low discriminative power
    """
    alpha = successes + 1
    beta = failures + 1
    n = alpha + beta
    return 4.0 * alpha * beta / (n * n)


@dataclass
class BayesianBatchSampler(BatchSampler[DataId, DataInst]):
    """Bayesian variance-based batch sampler for frontier targeting.

    Prioritizes samples where evaluation outcomes are most uncertain (balanced
    success/failure ratio). This differs from AdaBoost which focuses on "hard"
    samples - here we focus on "uncertain" samples that provide the strongest
    signal for distinguishing between candidates.

    Uses fractional counting for continuous scores: a score of 0.75 contributes
    0.75 to successes and 0.25 to failures. This preserves information from
    partial-credit evaluations (e.g., NYT Connections with 0.25/0.5/0.75/1.0)
    while remaining backward compatible with binary scores (0 and 1).

    Key behaviors:
        - Cold start (0, 0): max priority 1.0, explore unknown samples
        - Frontier (s ≈ f): max priority, samples where candidates disagree most
        - Confident (s >> f or f >> s): low priority, already determined

    Args:
        minibatch_size: Number of samples to return per batch.
        window: If set, only consider the most recent `window` trace entries when
            computing success/failure counts. None means use all history (default).
            This allows samples to regain priority when newer candidates can solve
            previously hard samples.
    """

    minibatch_size: int
    window: int | None = None  # sliding window for recent traces

    _outcome_history: dict[DataId, list[tuple[int, float]]] = field(default_factory=dict)
    _scores: dict[DataId, float] = field(default_factory=dict)
    _last_processed_trace_idx: int = field(default=-1)
    _last_sampled_ids: list[DataId] = field(default_factory=list)
    _residual_sampler: ResidualWeightedSampler | None = field(default=None)
    _cached_all_ids: list[DataId] = field(default_factory=list)

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample from empty loader.")

        self._update_counts_from_state(state)
        self._compute_scores(all_ids)
        self._normalize_scores(all_ids)
        self._update_residual_sampler(all_ids)
        return self._sample_from_residual(all_ids)

    def _update_counts_from_state(self, state: GEPAState) -> None:
        """Update outcome history based on new trace entries."""
        for trace_idx in range(self._last_processed_trace_idx + 1, len(state.full_program_trace)):
            trace = state.full_program_trace[trace_idx]
            subsample_ids = trace.get("subsample_ids")
            subsample_scores = trace.get("subsample_scores")

            if subsample_ids is None or subsample_scores is None:
                logger.debug(f"Trace {trace_idx} incomplete, stopping count update")
                break

            for data_id, score in zip(subsample_ids, subsample_scores, strict=False):
                clamped_score = max(0.0, min(1.0, float(score)))
                if data_id not in self._outcome_history:
                    self._outcome_history[data_id] = []
                self._outcome_history[data_id].append((trace_idx, clamped_score))

            self._last_processed_trace_idx = trace_idx

    def _get_windowed_counts(self, data_id: DataId) -> tuple[float, float]:
        """Get fractional success/failure counts within the window.

        Uses fractional counting where each score contributes proportionally:
        - score of 1.0 adds (1.0, 0.0) to (successes, failures)
        - score of 0.5 adds (0.5, 0.5) to (successes, failures)
        - score of 0.0 adds (0.0, 1.0) to (successes, failures)

        This preserves information from continuous scores (e.g., 0.25, 0.5, 0.75)
        while remaining backward compatible with binary scores (0 and 1).

        Args:
            data_id: The sample ID to get counts for.

        Returns:
            Tuple of (successes, failures) within the window.
        """
        history = self._outcome_history.get(data_id, [])
        if not history:
            return (0.0, 0.0)

        if self.window is None:
            # Use all history
            successes = sum(score for _, score in history)
            failures = sum(1.0 - score for _, score in history)
            return (successes, failures)

        # Only count entries within window
        cutoff = self._last_processed_trace_idx - self.window + 1
        successes = 0.0
        failures = 0.0
        for trace_idx, score in history:
            if trace_idx >= cutoff:
                successes += score
                failures += 1.0 - score
        return (successes, failures)

    def _compute_scores(self, all_ids: Sequence[DataId]) -> None:
        """Compute frontier scores for all samples."""
        for data_id in all_ids:
            s, f = self._get_windowed_counts(data_id)
            self._scores[data_id] = bayesian_frontier_score(s, f)

    def _normalize_scores(self, all_ids: Sequence[DataId]) -> None:
        """Normalize scores so they sum to the number of samples.

        This ensures average score = 1.0 for consistency with other samplers
        and easier visualization. Does not affect sampling behavior since
        ResidualWeightedSampler uses relative weights.
        """
        total = sum(self._scores[data_id] for data_id in all_ids)
        if total == 0:
            return
        scale = len(all_ids) / total
        for data_id in all_ids:
            self._scores[data_id] *= scale

    def _update_residual_sampler(self, all_ids: Sequence[DataId]) -> None:
        """Update residual sampler with current scores."""
        n = len(all_ids)

        # Check if we need to recreate the sampler (size changed or first time)
        if self._residual_sampler is None or len(self._cached_all_ids) != n or self._cached_all_ids != list(all_ids):
            self._residual_sampler = ResidualWeightedSampler(n)
            self._cached_all_ids = list(all_ids)

        # Update weights (scores are already in [0, 1])
        weights = [self._scores[data_id] for data_id in all_ids]
        self._residual_sampler.update_weights(weights)

    def _sample_from_residual(self, all_ids: Sequence[DataId]) -> list[DataId]:
        """Sample from residual sampler and convert indices to data IDs."""
        assert self._residual_sampler is not None

        k = min(self.minibatch_size, len(all_ids))
        indices = self._residual_sampler.sample(k)
        selected = [all_ids[idx] for idx in indices]

        self._last_sampled_ids = selected
        self._log_stats()
        return selected

    def _log_stats(self) -> None:
        """Log frontier score distribution statistics."""
        if self._scores:
            all_scores = list(self._scores.values())
            logger.debug(
                f"Frontier score stats: min={min(all_scores):.3f}, max={max(all_scores):.3f}, "
                f"mean={sum(all_scores) / len(all_scores):.3f}"
            )

    def get_scores(self) -> dict[DataId, float]:
        """Return current frontier scores (for debugging/inspection)."""
        return dict(self._scores)

    def get_counts(self) -> dict[DataId, tuple[float, float]]:
        """Return fractional (successes, failures) counts for each sample within the window."""
        return {data_id: self._get_windowed_counts(data_id) for data_id in self._outcome_history}

    def get_batch_weights(self) -> list[float] | None:
        """Return weights of samples in the most recent batch."""
        if not self._last_sampled_ids:
            return None
        return [self._scores.get(data_id, 1.0) for data_id in self._last_sampled_ids]

    def get_all_sample_weights(self) -> dict[DataId, float] | None:
        """Return per-sample frontier scores as {sample_id: score}.

        Returns:
            Dict with data_id: score for each sample,
            or None if no scores computed yet.
        """
        if not self._scores:
            return None
        return dict(self._scores)
