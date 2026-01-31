# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.logging import get_logger
from gepa.strategies.batch_sampler import BatchSampler

logger = get_logger()


@dataclass
class AdaBoostBatchSampler(BatchSampler[DataId, DataInst]):
    """AdaBoost-style weighted sampling for mini-batch selection.

    Each sample has a weight that is updated based on evaluation scores:
    - Samples with low scores (errors) have their weights increased
    - Samples with high scores have their weights decreased
    - Weights are used for probabilistic sampling without replacement
    """

    minibatch_size: int
    beta: float = 1.0  # Weight update sensitivity
    min_weight: float = 0.1  # Prevent weights from going to zero
    max_weight: float = 10.0  # Prevent weight explosion
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    _weights: dict[DataId, float] = field(default_factory=dict)
    _last_processed_trace_idx: int = field(default=-1)
    _last_sampled_ids: list[DataId] = field(default_factory=list)

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample from empty loader.")

        self._initialize_weights(all_ids)
        self._update_weights_from_state(state)
        self._normalize_weights(all_ids)
        return self._weighted_sample_without_replacement(all_ids)

    def _initialize_weights(self, all_ids: Sequence[DataId]) -> None:
        """Initialize weights for any new data IDs."""
        for data_id in all_ids:
            if data_id not in self._weights:
                self._weights[data_id] = 1.0

    def _update_weights_from_state(self, state: GEPAState) -> None:
        """Update weights based on new trace entries since last processing."""
        for trace_idx in range(self._last_processed_trace_idx + 1, len(state.full_program_trace)):
            trace = state.full_program_trace[trace_idx]
            subsample_ids = trace.get("subsample_ids")
            subsample_scores = trace.get("subsample_scores")

            if subsample_ids is None or subsample_scores is None:
                # Current trace incomplete, stop here and retry next time
                logger.debug(f"Trace {trace_idx} incomplete, stopping weight update")
                break

            for data_id, score in zip(subsample_ids, subsample_scores, strict=False):
                self._update_single_weight(data_id, score)

            self._last_processed_trace_idx = trace_idx

    def _update_single_weight(self, data_id: DataId, score: float) -> None:
        """Update weight for a single sample based on its score.

        Uses formula: new_weight = old_weight * exp(beta * (error - 0.5))
        where error = 1.0 - score.

        - error > 0.5 (score < 0.5) -> multiplier > 1 -> weight increases
        - error < 0.5 (score > 0.5) -> multiplier < 1 -> weight decreases
        """
        score = max(0.0, min(1.0, score))
        error = 1.0 - score
        multiplier = math.exp(self.beta * (error - 0.5))
        new_weight = self._weights.get(data_id, 1.0) * multiplier
        self._weights[data_id] = max(self.min_weight, min(self.max_weight, new_weight))

    def _normalize_weights(self, all_ids: Sequence[DataId]) -> None:
        """Normalize weights so they sum to the number of samples."""
        total = sum(self._weights[data_id] for data_id in all_ids)
        if total == 0:
            return
        scale = len(all_ids) / total
        for data_id in all_ids:
            self._weights[data_id] *= scale

    def _weighted_sample_without_replacement(self, all_ids: Sequence[DataId]) -> list[DataId]:
        """Sample minibatch_size items without replacement using weights."""
        n = min(self.minibatch_size, len(all_ids))
        remaining = list(all_ids)
        remaining_weights = [self._weights[data_id] for data_id in remaining]
        selected: list[DataId] = []

        for _ in range(n):
            if sum(remaining_weights) == 0:
                break
            idx = self.rng.choices(range(len(remaining)), weights=remaining_weights, k=1)[0]
            selected.append(remaining.pop(idx))
            remaining_weights.pop(idx)

        self._last_sampled_ids = selected
        self._log_weight_stats()
        return selected

    def _log_weight_stats(self) -> None:
        """Log weight distribution statistics for debugging."""
        all_weights = list(self._weights.values())
        if all_weights:
            logger.debug(
                f"Weight stats: min={min(all_weights):.3f}, max={max(all_weights):.3f}, "
                f"mean={sum(all_weights) / len(all_weights):.3f}"
            )

    def get_weights(self) -> dict[DataId, float]:
        """Return current weights (for debugging/inspection)."""
        return dict(self._weights)

    def get_last_sampled_avg_weight(self) -> float:
        """Return the average weight of the most recently sampled batch."""
        if not self._last_sampled_ids:
            return 1.0
        weights = [self._weights.get(data_id, 1.0) for data_id in self._last_sampled_ids]
        return sum(weights) / len(weights)

    def get_train_sample_weight_stats(self) -> dict[str, float] | None:
        """Return weight statistics for all training samples.

        Returns:
            Dict with train_sample_weight_avg, train_sample_weight_max, train_sample_weight_min,
            or None if no weights initialized yet.
        """
        if not self._weights:
            return None
        weights = list(self._weights.values())
        return {
            "train_sample_weight_avg": sum(weights) / len(weights),
            "train_sample_weight_max": max(weights),
            "train_sample_weight_min": min(weights),
        }


@dataclass
class PMaxBatchSampler(BatchSampler[DataId, DataInst]):
    """AdaBoost with max-score reset sampling.

    Tracks the best score seen for each training sample internally and uses it
    to decide how to update weights:

    Key behavior:
    - Unattempted samples: weight = unattempted_boost (default 1.5)
      - Encourage exploration of unseen samples
    - Attempted but never-solved samples (best_score == 0): AdaBoost weight update
      - May be difficult, boost weight to get more training opportunities
    - Once-solved samples (best_score > 0): weight reset to 1.0
      - Already proven solvable, no need for extra focus
    """

    minibatch_size: int
    beta: float = 1.0  # Weight update sensitivity
    min_weight: float = 0.1  # Prevent weights from going to zero
    max_weight: float = 10.0  # Prevent weight explosion
    unattempted_boost: float = 1.5  # Initial weight boost for unattempted samples
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    _weights: dict[DataId, float] = field(default_factory=dict)
    _best_scores: dict[DataId, float] = field(default_factory=dict)  # Track best score per training sample
    _attempted: set[DataId] = field(default_factory=set)  # Track which samples have been attempted
    _last_processed_trace_idx: int = field(default=-1)
    _last_sampled_ids: list[DataId] = field(default_factory=list)

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample from empty loader.")

        self._initialize_weights(all_ids)
        self._update_weights_from_state(state)
        self._normalize_weights(all_ids)
        return self._weighted_sample_without_replacement(all_ids)

    def _initialize_weights(self, all_ids: Sequence[DataId]) -> None:
        """Initialize weights for any new data IDs with unattempted_boost."""
        for data_id in all_ids:
            if data_id not in self._weights:
                self._weights[data_id] = self.unattempted_boost

    def _update_weights_from_state(self, state: GEPAState) -> None:
        """Update weights based on new trace entries since last processing.

        Uses PMax logic:
        - Unattempted samples: keep weight at unattempted_boost
        - Attempted but never-solved (best_score == 0): AdaBoost weight update
        - Once-solved (best_score > 0): reset weight to 1.0

        Tracks best scores internally from training set evaluations.
        """
        for trace_idx in range(self._last_processed_trace_idx + 1, len(state.full_program_trace)):
            trace = state.full_program_trace[trace_idx]
            subsample_ids = trace.get("subsample_ids")
            subsample_scores = trace.get("subsample_scores")

            if subsample_ids is None or subsample_scores is None:
                # Current trace incomplete, stop here and retry next time
                logger.debug(f"Trace {trace_idx} incomplete, stopping weight update")
                break

            solved_count = 0
            unsolved_count = 0
            for data_id, score in zip(subsample_ids, subsample_scores, strict=False):
                # Mark as attempted
                first_attempt = data_id not in self._attempted
                self._attempted.add(data_id)

                # Update best score tracking
                current_best = self._best_scores.get(data_id, 0.0)
                if score > current_best:
                    self._best_scores[data_id] = score

                # Use the updated best score for weight decision
                best_score = self._best_scores.get(data_id, 0.0)
                if best_score > 0:
                    # Once-solved sample: reset weight to 1.0
                    self._weights[data_id] = 1.0
                    solved_count += 1
                else:
                    # Never-solved sample: AdaBoost weight update
                    # On first attempt, reset from unattempted_boost to 1.0 before AdaBoost
                    if first_attempt:
                        self._weights[data_id] = 1.0
                    self._update_single_weight(data_id, score)
                    unsolved_count += 1

            logger.debug(
                f"Processed trace {trace_idx}: {solved_count} solved (weights reset to 1.0), "
                f"{unsolved_count} unsolved (AdaBoost update)"
            )
            self._last_processed_trace_idx = trace_idx

    def _update_single_weight(self, data_id: DataId, score: float) -> None:
        """Update weight for a single never-solved sample based on its score.

        Uses formula: new_weight = old_weight * exp(beta * (error - 0.5))
        where error = 1.0 - score.

        - error > 0.5 (score < 0.5) -> multiplier > 1 -> weight increases
        - error < 0.5 (score > 0.5) -> multiplier < 1 -> weight decreases
        """
        score = max(0.0, min(1.0, score))
        error = 1.0 - score
        multiplier = math.exp(self.beta * (error - 0.5))
        new_weight = self._weights.get(data_id, 1.0) * multiplier
        self._weights[data_id] = max(self.min_weight, min(self.max_weight, new_weight))

    def _normalize_weights(self, all_ids: Sequence[DataId]) -> None:
        """Normalize weights so they sum to the number of samples.

        Before normalization, re-apply unattempted_boost to unattempted samples
        to ensure their relative weight advantage is preserved.
        """
        # Re-apply unattempted_boost to preserve relative weight
        for data_id in all_ids:
            if data_id not in self._attempted:
                self._weights[data_id] = self.unattempted_boost

        total = sum(self._weights[data_id] for data_id in all_ids)
        if total == 0:
            return
        scale = len(all_ids) / total
        for data_id in all_ids:
            self._weights[data_id] *= scale

    def _weighted_sample_without_replacement(self, all_ids: Sequence[DataId]) -> list[DataId]:
        """Sample minibatch_size items without replacement using weights."""
        n = min(self.minibatch_size, len(all_ids))
        remaining = list(all_ids)
        remaining_weights = [self._weights[data_id] for data_id in remaining]
        selected: list[DataId] = []

        for _ in range(n):
            if sum(remaining_weights) == 0:
                break
            idx = self.rng.choices(range(len(remaining)), weights=remaining_weights, k=1)[0]
            selected.append(remaining.pop(idx))
            remaining_weights.pop(idx)

        self._last_sampled_ids = selected
        self._log_weight_stats()
        return selected

    def _log_weight_stats(self) -> None:
        """Log weight distribution statistics for debugging."""
        all_weights = list(self._weights.values())
        if all_weights:
            logger.debug(
                f"Weight stats: min={min(all_weights):.3f}, max={max(all_weights):.3f}, "
                f"mean={sum(all_weights) / len(all_weights):.3f}"
            )

    def get_weights(self) -> dict[DataId, float]:
        """Return current weights (for debugging/inspection)."""
        return dict(self._weights)

    def get_last_sampled_avg_weight(self) -> float:
        """Return the average weight of the most recently sampled batch."""
        if not self._last_sampled_ids:
            return 1.0
        weights = [self._weights.get(data_id, 1.0) for data_id in self._last_sampled_ids]
        return sum(weights) / len(weights)

    def get_train_sample_weight_stats(self) -> dict[str, float] | None:
        """Return weight statistics for all training samples.

        Returns:
            Dict with train_sample_weight_avg, train_sample_weight_max, train_sample_weight_min,
            or None if no weights initialized yet.
        """
        if not self._weights:
            return None
        all_weights = list(self._weights.values())
        return {
            "train_sample_weight_avg": sum(all_weights) / len(all_weights),
            "train_sample_weight_max": max(all_weights),
            "train_sample_weight_min": min(all_weights),
        }
