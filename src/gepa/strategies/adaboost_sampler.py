# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import BatchSampler


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
                continue

            for data_id, score in zip(subsample_ids, subsample_scores, strict=False):
                self._update_single_weight(data_id, score)

        self._last_processed_trace_idx = len(state.full_program_trace) - 1

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
        return selected

    def get_weights(self) -> dict[DataId, float]:
        """Return current weights (for debugging/inspection)."""
        return dict(self._weights)

    def get_last_sampled_avg_weight(self) -> float:
        """Return the average weight of the most recently sampled batch."""
        if not self._last_sampled_ids:
            return 1.0
        weights = [self._weights.get(data_id, 1.0) for data_id in self._last_sampled_ids]
        return sum(weights) / len(weights)
