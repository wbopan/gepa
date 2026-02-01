# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.residual_weighted_sampler import ResidualWeightedSampler


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(),
            state.per_program_tracked_scores,
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return idxmax(state.program_full_scores_val_set)


class EpsilonGreedyCandidateSelector(CandidateSelector):
    def __init__(self, epsilon: float, rng: random.Random | None):
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(state.program_candidates) - 1)
        else:
            return idxmax(state.program_full_scores_val_set)


class AvgFamilyScoreCandidateSelector(CandidateSelector):
    """Select candidate based on average family score.

    Weight formula: W = (Score_parent + Σ Score_children) / (1 + |children|)

    This is the optimal strategy under zero-prior constraint:
    - When no children exist, weight = parent score (minimal reasonable prior)
    - As children are produced, weight is updated by actual data
    """

    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        scores = state.per_program_tracked_scores
        n = len(state.program_candidates)

        family_scores = []
        for parent_idx in range(n):
            parent_score = scores[parent_idx]
            children_indices = [
                i for i, parents in enumerate(state.parent_program_for_candidate) if parent_idx in parents
            ]
            children_scores = [scores[i] for i in children_indices]
            family_score = (parent_score + sum(children_scores)) / (1 + len(children_scores))
            family_scores.append(family_score)

        return idxmax(family_scores)


class MaxFamilyScoreCandidateSelector(CandidateSelector):
    """Select candidate using max family score with residual weighted sampling.

    Family score = max(parent_score, max(children_scores))

    Unlike AvgFamilyScoreCandidateSelector which punishes good candidates for
    having high-variance children, this selector rewards candidates for their
    best offspring performance.

    Uses normalized power transform for selection pressure:
        weight = score^power / sum(scores^power)

    This ensures:
        - High scorers are selected much more often (0.8 vs 0.4 = 8x with power=3)
        - But low scorers still get occasional chances (exploration)
        - Weights sum to 1.0, matching k=1 sampling rate

    Uses ResidualWeightedSampler to ensure all candidates get eventual
    selection opportunities proportional to their family scores, avoiding
    degenerate greedy selection.

    Args:
        power: Exponent for selection pressure. Higher = more greedy.
            - power=2: Mild pressure (0.8 vs 0.4 = 4x)
            - power=3: Balanced (0.8 vs 0.4 = 8x) [default]
            - power=4: Aggressive (0.8 vs 0.4 = 16x)
    """

    def __init__(self, power: float = 3.0):
        self._sampler: ResidualWeightedSampler | None = None
        self._power = power

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        scores = state.per_program_tracked_scores
        n = len(state.program_candidates)

        # Compute max family scores
        family_scores = []
        for parent_idx in range(n):
            parent_score = scores[parent_idx]
            children_indices = [
                i for i, parents in enumerate(state.parent_program_for_candidate) if parent_idx in parents
            ]
            children_scores = [scores[i] for i in children_indices]

            if children_scores:
                family_score = max(parent_score, max(children_scores))
            else:
                family_score = parent_score
            family_scores.append(family_score)

        # Apply power transform for selection pressure
        raw_weights = [max(0.0, s) ** self._power for s in family_scores]

        # Normalize to sum=1.0 (required for k=1 sampling rate)
        total_weight = sum(raw_weights)
        if total_weight < 1e-9:
            # All scores near zero: fall back to uniform
            weights = [1.0 / n] * n
        else:
            weights = [w / total_weight for w in raw_weights]

        # Initialize sampler on first call; update_weights handles extension automatically
        if self._sampler is None:
            self._sampler = ResidualWeightedSampler(n)

        # Update weights (auto-extends if n grew) and sample
        self._sampler.update_weights(weights)
        return self._sampler.sample(k=1)[0]
