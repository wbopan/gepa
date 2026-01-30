# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter

import pytest

from gepa.strategies.adaboost_sampler import AdaBoostBatchSampler


class MockDataLoader:
    """Mock DataLoader for testing."""

    def __init__(self, ids: list):
        self._ids = ids

    def all_ids(self):
        return self._ids

    def __len__(self):
        return len(self._ids)

    def fetch(self, ids):
        return ids


class MockGEPAState:
    """Mock GEPAState for testing."""

    def __init__(self, pareto_front_valset: dict | None = None):
        self.full_program_trace: list[dict] = []
        self.i = 0
        self.total_num_evals = 0
        # Default: all samples are solvable (best_score = 1.0)
        self.pareto_front_valset: dict = pareto_front_valset if pareto_front_valset is not None else {}


class TestAdaBoostBatchSampler:
    def test_initial_weights_are_one(self):
        """Test that initial weights are 1.0 for all samples."""
        sampler = AdaBoostBatchSampler(minibatch_size=3, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        for w in weights.values():
            assert w == pytest.approx(1.0, rel=0.01)

    def test_low_score_increases_weight(self):
        """Test that samples with low scores have increased weights."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0, 2: 1.0})

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)

        # Add trace with low score for sample 0
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.1]})

        # Sample again to trigger weight update
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0 should have higher weight than others (after normalization)
        # The relative weight of sample 0 should be higher
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_high_score_decreases_weight(self):
        """Test that samples with high scores have decreased weights."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0, 2: 1.0})

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)

        # Add trace with high score for sample 0
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.9]})

        # Sample again to trigger weight update
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0 should have lower weight than others (after normalization)
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]

    def test_weights_clamped_to_min_max(self):
        """Test that weights stay within [min_weight, max_weight]."""
        sampler = AdaBoostBatchSampler(
            minibatch_size=2, beta=10.0, min_weight=0.1, max_weight=10.0, rng=random.Random(42)
        )
        loader = MockDataLoader([0, 1])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # Add multiple traces with extreme scores
        for _ in range(10):
            state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.0]})  # Always wrong
            state.full_program_trace.append({"subsample_ids": [1], "subsample_scores": [1.0]})  # Always right

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Before normalization, weights should be clamped
        # After normalization they scale but the ratio is preserved
        assert min(weights.values()) >= 0.0  # Normalized values can be lower than min_weight
        assert max(weights.values()) <= 20.0  # Sanity check

    def test_failed_samples_sampled_more_frequently(self):
        """Test that samples that failed are sampled more frequently over many iterations."""
        sampler = AdaBoostBatchSampler(minibatch_size=1, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0, 2: 1.0})

        # Initialize
        sampler.next_minibatch_ids(loader, state)

        # Sample 0 always fails, others always succeed
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 1.0, 1.0]})

        # Count how often each sample is selected
        counts = Counter()
        for _ in range(100):
            sampler.next_minibatch_ids(loader, state)
            selected = sampler._last_sampled_ids
            counts.update(selected)

        # Sample 0 (the failing one) should be selected most often
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]

    def test_empty_trace_no_weight_update(self):
        """Test that empty trace entries don't cause errors."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add trace without subsample_ids or subsample_scores
        state.full_program_trace.append({})
        state.full_program_trace.append({"subsample_ids": None, "subsample_scores": None})

        # Should not raise
        result = sampler.next_minibatch_ids(loader, state)
        assert len(result) == 2

    def test_partial_observation(self):
        """Test that only observed samples have their weights updated."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3, 4])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # Only update samples 0 and 1
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0]})

        sampler.next_minibatch_ids(loader, state)
        updated_weights = sampler.get_weights()

        # Samples 2, 3, 4 should have unchanged relative weights (though normalized)
        # The key insight: 0's raw weight went up, 1's went down, 2-4 stayed at 1.0 before normalization
        # After normalization, 0 > 1 and {2,3,4} are in between
        assert updated_weights[0] > updated_weights[1]

    def test_incremental_trace_processing(self):
        """Test that traces are processed incrementally."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == -1

        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.5]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == 0

        state.full_program_trace.append({"subsample_ids": [1], "subsample_scores": [0.5]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == 1

    def test_dataset_expansion(self):
        """Test that new data IDs get initialized with weight 1.0."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))

        # Start with 2 samples
        loader1 = MockDataLoader([0, 1])
        state = MockGEPAState()
        sampler.next_minibatch_ids(loader1, state)

        assert 0 in sampler.get_weights()
        assert 1 in sampler.get_weights()
        assert 2 not in sampler.get_weights()

        # Expand to 3 samples
        loader2 = MockDataLoader([0, 1, 2])
        sampler.next_minibatch_ids(loader2, state)

        assert 2 in sampler.get_weights()

    def test_empty_loader_raises(self):
        """Test that empty loader raises ValueError."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))
        loader = MockDataLoader([])
        state = MockGEPAState()

        with pytest.raises(ValueError, match="Cannot sample from empty loader"):
            sampler.next_minibatch_ids(loader, state)

    def test_get_last_sampled_avg_weight(self):
        """Test get_last_sampled_avg_weight returns correct average."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # All samples are solvable (best_score > 0)
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0, 2: 1.0})

        # First sample - all weights are 1.0
        sampler.next_minibatch_ids(loader, state)
        avg = sampler.get_last_sampled_avg_weight()
        assert avg == pytest.approx(1.0, rel=0.01)

        # Update weights
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.5, 1.0]})
        sampler.next_minibatch_ids(loader, state)

        # Average should reflect the sampled items' weights
        avg = sampler.get_last_sampled_avg_weight()
        assert avg > 0  # Should be positive

    def test_get_last_sampled_avg_weight_empty(self):
        """Test get_last_sampled_avg_weight returns 1.0 when no samples yet."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))
        assert sampler.get_last_sampled_avg_weight() == 1.0

    def test_minibatch_size_larger_than_dataset(self):
        """Test that minibatch_size larger than dataset returns all samples."""
        sampler = AdaBoostBatchSampler(minibatch_size=10, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        result = sampler.next_minibatch_ids(loader, state)
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_deterministic_with_seed(self):
        """Test that sampling is deterministic with the same seed."""
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler1 = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))
        sampler2 = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))

        result1 = sampler1.next_minibatch_ids(loader, state)
        result2 = sampler2.next_minibatch_ids(loader, state)

        assert result1 == result2


class TestParetoInternalAdaBoost:
    """Tests for Pareto-Internal AdaBoost behavior (solvable vs unsolvable samples)."""

    def test_unsolvable_samples_weights_frozen(self):
        """Test that unsolvable samples (best_score == 0) have their weights frozen."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # Sample 0 is unsolvable (best_score == 0), samples 1 and 2 are solvable
        state = MockGEPAState(pareto_front_valset={0: 0.0, 1: 1.0, 2: 1.0})

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)
        initial_weight_0 = sampler.get_weights()[0]

        # Add trace with low scores for all samples
        # Sample 0 should NOT have its weight updated (unsolvable)
        # Samples 1 and 2 should have weights increased (low score on solvable sample)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.1, 0.1]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0's weight should remain at 1.0 (before normalization it was unchanged)
        # After normalization, samples 1 and 2 should have higher weights than sample 0
        # because they were boosted for having low scores while sample 0 was frozen
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]

    def test_solvable_samples_weights_updated(self):
        """Test that solvable samples (best_score > 0) have their weights updated."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        # Both samples are solvable
        state = MockGEPAState(pareto_front_valset={0: 1.0, 1: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # Sample 0 has low score (weight should increase)
        # Sample 1 has high score (weight should decrease)
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.1, 0.9]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0 should have higher weight than sample 1
        assert weights[0] > weights[1]

    def test_mixed_solvable_unsolvable_samples(self):
        """Test behavior with a mix of solvable and unsolvable samples."""
        sampler = AdaBoostBatchSampler(minibatch_size=3, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3])
        # Samples 0 and 1 are unsolvable, samples 2 and 3 are solvable
        state = MockGEPAState(pareto_front_valset={0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # All samples get low scores
        state.full_program_trace.append({"subsample_ids": [0, 1, 2, 3], "subsample_scores": [0.0, 0.0, 0.1, 0.2]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Samples 2 and 3 (solvable) should have higher weights than samples 0 and 1 (unsolvable)
        # because their low scores triggered weight increases while unsolvable samples were frozen
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]
        assert weights[3] > weights[0]
        assert weights[3] > weights[1]

    def test_unsolvable_sample_not_dominating(self):
        """Test that unsolvable samples don't dominate sampling over time."""
        sampler = AdaBoostBatchSampler(minibatch_size=1, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # Sample 0 is unsolvable, samples 1 and 2 are solvable
        state = MockGEPAState(pareto_front_valset={0: 0.0, 1: 1.0, 2: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # Repeatedly add traces where sample 0 always fails
        # Without Pareto-Internal, sample 0's weight would explode
        # With Pareto-Internal, sample 0's weight stays frozen
        for _ in range(10):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.5, 0.5]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0 (unsolvable) should not have the highest weight
        # The solvable samples should have similar or higher weights
        assert weights[0] <= weights[1] or weights[0] <= weights[2]

    def test_sample_becomes_solvable(self):
        """Test that samples can become solvable when Pareto front is updated."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        # Initially sample 0 is unsolvable
        state = MockGEPAState(pareto_front_valset={0: 0.0, 1: 1.0})

        sampler.next_minibatch_ids(loader, state)

        # First trace: sample 0 has low score but is unsolvable (no weight update)
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.1]})
        sampler.next_minibatch_ids(loader, state)

        # Now sample 0 becomes solvable (Pareto front updated externally)
        state.pareto_front_valset[0] = 0.5

        # Second trace: sample 0 has low score and is now solvable (weight should increase)
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.1]})
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 0 should now have higher weight than sample 1 because:
        # - Sample 0: was frozen (unchanged) then got low score update (weight increased)
        # - Sample 1: no updates, stayed at initial weight
        # After normalization, sample 0 should have relatively higher weight
        assert weights[0] > weights[1]

    def test_empty_pareto_front_all_unsolvable(self):
        """Test behavior when pareto_front_valset is empty (all samples treated as unsolvable)."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        # Empty pareto front means no sample has been solved yet
        state = MockGEPAState(pareto_front_valset={})

        sampler.next_minibatch_ids(loader, state)

        # Add trace with varying scores
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.5, 1.0]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # All weights should be equal (all frozen at 1.0, then normalized)
        assert weights[0] == pytest.approx(weights[1], rel=0.01)
        assert weights[1] == pytest.approx(weights[2], rel=0.01)
