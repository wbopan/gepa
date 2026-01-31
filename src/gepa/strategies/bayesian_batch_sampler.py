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


def bayesian_frontier_score(successes: int, failures: int) -> float:
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

    Key behaviors:
        - Cold start (0, 0): max priority 1.0, explore unknown samples
        - Frontier (s ≈ f): max priority, samples where candidates disagree most
        - Confident (s >> f or f >> s): low priority, already determined
    """

    minibatch_size: int
    binarize_threshold: float = 0.5  # score >= threshold counts as success

    _successes: dict[DataId, int] = field(default_factory=dict)
    _failures: dict[DataId, int] = field(default_factory=dict)
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
        self._update_residual_sampler(all_ids)
        return self._sample_from_residual(all_ids)

    def _update_counts_from_state(self, state: GEPAState) -> None:
        """Update success/failure counts based on new trace entries."""
        for trace_idx in range(self._last_processed_trace_idx + 1, len(state.full_program_trace)):
            trace = state.full_program_trace[trace_idx]
            subsample_ids = trace.get("subsample_ids")
            subsample_scores = trace.get("subsample_scores")

            if subsample_ids is None or subsample_scores is None:
                logger.debug(f"Trace {trace_idx} incomplete, stopping count update")
                break

            for data_id, score in zip(subsample_ids, subsample_scores, strict=False):
                if score >= self.binarize_threshold:
                    self._successes[data_id] = self._successes.get(data_id, 0) + 1
                else:
                    self._failures[data_id] = self._failures.get(data_id, 0) + 1

            self._last_processed_trace_idx = trace_idx

    def _compute_scores(self, all_ids: Sequence[DataId]) -> None:
        """Compute frontier scores for all samples."""
        for data_id in all_ids:
            s = self._successes.get(data_id, 0)
            f = self._failures.get(data_id, 0)
            self._scores[data_id] = bayesian_frontier_score(s, f)

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

    def get_counts(self) -> dict[DataId, tuple[int, int]]:
        """Return (successes, failures) counts for each sample."""
        all_ids = set(self._successes.keys()) | set(self._failures.keys())
        return {data_id: (self._successes.get(data_id, 0), self._failures.get(data_id, 0)) for data_id in all_ids}

    def get_last_sampled_avg_weight(self) -> float:
        """Return the average frontier score of the most recently sampled batch."""
        if not self._last_sampled_ids:
            return 1.0
        scores = [self._scores.get(data_id, 1.0) for data_id in self._last_sampled_ids]
        return sum(scores) / len(scores)

    def get_train_sample_weight_stats(self) -> dict[str, float] | None:
        """Return frontier score statistics for all training samples.

        Returns:
            Dict with train/frontier_score_avg, train/frontier_score_max, train/frontier_score_min,
            or None if no scores computed yet.
        """
        if not self._scores:
            return None
        all_scores = list(self._scores.values())
        return {
            "train/frontier_score_avg": sum(all_scores) / len(all_scores),
            "train/frontier_score_max": max(all_scores),
            "train/frontier_score_min": min(all_scores),
        }
