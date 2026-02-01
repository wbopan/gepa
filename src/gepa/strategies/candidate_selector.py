# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


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
