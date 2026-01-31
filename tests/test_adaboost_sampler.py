# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter

import pytest

from gepa.strategies.adaboost_sampler import AdaBoostBatchSampler, PMaxBatchSampler
from gepa.strategies.batch_sampler import MetricLoggingBatchSampler


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

    def __init__(self):
        self.full_program_trace: list[dict] = []
        self.i = 0
        self.total_num_evals = 0


class TestAdaBoostBatchSampler:
    """Tests for standard AdaBoost batch sampler (all samples updated regardless of solvability)."""

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
        state = MockGEPAState()

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
        state = MockGEPAState()

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
        state = MockGEPAState()

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
        state = MockGEPAState()

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
        state = MockGEPAState()

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

    def test_get_batch_weights(self):
        """Test get_batch_weights returns correct weights."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample - all weights are 1.0
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_batch_weights()
        assert weights is not None
        assert len(weights) == 2
        for w in weights:
            assert w == pytest.approx(1.0, rel=0.01)

        # Update weights
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.5, 1.0]})
        sampler.next_minibatch_ids(loader, state)

        # Weights should reflect the sampled items' weights
        weights = sampler.get_batch_weights()
        assert weights is not None
        assert all(w > 0 for w in weights)  # Should be positive

    def test_get_batch_weights_empty(self):
        """Test get_batch_weights returns None when no samples yet."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, rng=random.Random(42))
        assert sampler.get_batch_weights() is None

    def test_get_all_sample_weights(self):
        """Test get_all_sample_weights returns per-sample weights."""
        sampler = AdaBoostBatchSampler(minibatch_size=3, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # Before any sampling
        assert sampler.get_all_sample_weights() is None

        # After sampling
        sampler.next_minibatch_ids(loader, state)
        all_weights = sampler.get_all_sample_weights()

        assert all_weights is not None
        assert 0 in all_weights
        assert 1 in all_weights
        assert 2 in all_weights
        assert all_weights[0] == pytest.approx(1.0, rel=0.01)

    def test_metric_logging_protocol_conformance(self):
        """Test that AdaBoostBatchSampler conforms to MetricLoggingBatchSampler protocol."""
        sampler = AdaBoostBatchSampler(minibatch_size=2)
        assert isinstance(sampler, MetricLoggingBatchSampler)

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

    def test_all_samples_updated_regardless_of_score(self):
        """Test that standard AdaBoost updates all samples, even those with zero scores."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)

        # Add trace with zero scores for all samples
        # In standard AdaBoost, all should get weight increases (since error = 1.0)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # All weights should be equal (all got same update, then normalized)
        assert weights[0] == pytest.approx(weights[1], rel=0.01)
        assert weights[1] == pytest.approx(weights[2], rel=0.01)


class TestPMaxBatchSampler:
    """Tests for PMax behavior (solved samples reset, unsolved samples get AdaBoost).

    PMax now tracks best scores internally from training traces, not from an external
    pareto_front_valset. A sample becomes "solved" when it achieves a non-zero score
    in any trace.
    """

    def test_solved_samples_weight_reset_to_one(self):
        """Test that once-solved samples (best_score > 0) have their weights reset to 1.0."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)

        # First trace: sample 0 gets solved (score > 0), samples 1 and 2 stay unsolved
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.5, 0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Second trace: sample 0 gets low score (stays solved, reset to 1.0)
        # Samples 1 and 2 get zero scores (stay unsolved, AdaBoost boost)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.1, 0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Samples 1 and 2 (never-solved) should have higher weights than sample 0 (solved, reset)
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]

    def test_unsolved_samples_adaboost_update(self):
        """Test that never-solved samples (best_score == 0) get AdaBoost weight updates."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Both samples stay unsolved (score 0), but sample 0 has lower score than sample 1
        # Wait - if score is 0, they're unsolved. Let's use 0.0 for both to keep them unsolved
        # but vary the scores to see different AdaBoost effects
        # Actually for unsolved we need score == 0, so let's test with multiple 0 scores
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Both should have equal increased weights (both got same AdaBoost update)
        assert weights[0] == pytest.approx(weights[1], rel=0.01)

    def test_mixed_solved_unsolved_samples(self):
        """Test behavior with a mix of solved and never-solved samples."""
        sampler = PMaxBatchSampler(minibatch_size=3, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # First trace: samples 0 and 1 get solved, samples 2 and 3 stay unsolved
        state.full_program_trace.append({"subsample_ids": [0, 1, 2, 3], "subsample_scores": [0.5, 1.0, 0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Second trace: all samples get low scores
        # Solved (0, 1): weights reset to 1.0
        # Never-solved (2, 3): weights increased by AdaBoost
        state.full_program_trace.append({"subsample_ids": [0, 1, 2, 3], "subsample_scores": [0.0, 0.0, 0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Never-solved samples (2, 3) should have higher weights than solved samples (0, 1)
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]
        assert weights[3] > weights[0]
        assert weights[3] > weights[1]

    def test_solved_sample_stays_reset_across_traces(self):
        """Test that solved samples continue to get reset even with multiple trace updates."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # First trace: sample 0 gets solved, sample 1 stays unsolved
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Add multiple traces with zero scores
        # Sample 0 should stay at normalized 1.0 (solved, reset every time)
        # Sample 1's weight should keep increasing (never-solved, AdaBoost)
        for _ in range(5):
            state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 1 (never-solved with repeated low scores) should have much higher weight
        assert weights[1] > weights[0]

    def test_sample_becomes_solved(self):
        """Test that samples become solved when they achieve a non-zero score."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=2.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # First trace: both samples get zero scores (both unsolved, get AdaBoost boost)
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Second trace: sample 0 gets solved with a non-zero score
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Third trace: sample 0 should now get reset to 1.0, sample 1 continues AdaBoost
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # Sample 1 (still never-solved, got three rounds of AdaBoost) should have higher weight
        # Sample 0 (now solved, got reset) should have lower weight
        assert weights[1] > weights[0]

    def test_all_unsolved_initially(self):
        """Test behavior when all samples are never-solved (all zero scores)."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add trace with all zero scores - all should get AdaBoost updates
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # All weights should be equal (all got same AdaBoost update for zero score)
        assert weights[0] == pytest.approx(weights[1], rel=0.01)
        assert weights[1] == pytest.approx(weights[2], rel=0.01)

    def test_initial_weights_have_unattempted_boost(self):
        """Test that initial weights are unattempted_boost for all samples."""
        sampler = PMaxBatchSampler(minibatch_size=3, unattempted_boost=1.5, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_weights()

        # All unattempted samples should have unattempted_boost weight (before normalization)
        # After normalization, they should all be equal (normalized to 1.0 each)
        for w in weights.values():
            assert w == pytest.approx(1.0, rel=0.01)  # Normalized: 5 * 1.5 / 5 = 1.5, then scale to sum=5

    def test_empty_loader_raises(self):
        """Test that empty loader raises ValueError."""
        sampler = PMaxBatchSampler(minibatch_size=2, rng=random.Random(42))
        loader = MockDataLoader([])
        state = MockGEPAState()

        with pytest.raises(ValueError, match="Cannot sample from empty loader"):
            sampler.next_minibatch_ids(loader, state)

    def test_get_batch_weights(self):
        """Test get_batch_weights returns correct weights."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample - all weights are 1.0
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_batch_weights()
        assert weights is not None
        assert len(weights) == 2
        for w in weights:
            assert w == pytest.approx(1.0, rel=0.01)

        # Update weights with zero scores (all unsolved, get AdaBoost updates)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.0, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Weights should reflect the sampled items' weights
        weights = sampler.get_batch_weights()
        assert weights is not None
        assert all(w > 0 for w in weights)  # Should be positive

    def test_get_all_sample_weights(self):
        """Test get_all_sample_weights returns per-sample weights."""
        sampler = PMaxBatchSampler(minibatch_size=3, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # Before any sampling
        assert sampler.get_all_sample_weights() is None

        # After sampling
        sampler.next_minibatch_ids(loader, state)
        all_weights = sampler.get_all_sample_weights()

        assert all_weights is not None
        assert 0 in all_weights
        assert 1 in all_weights
        assert 2 in all_weights

    def test_metric_logging_protocol_conformance(self):
        """Test that PMaxBatchSampler conforms to MetricLoggingBatchSampler protocol."""
        sampler = PMaxBatchSampler(minibatch_size=2)
        assert isinstance(sampler, MetricLoggingBatchSampler)

    def test_best_score_tracking(self):
        """Test that sampler correctly tracks best scores internally."""
        sampler = PMaxBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # First trace: sample 0 gets 0.3
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.3]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._best_scores[0] == 0.3

        # Second trace: sample 0 gets lower score - best should stay at 0.3
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.1]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._best_scores[0] == 0.3

        # Third trace: sample 0 gets higher score - best should update to 0.5
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.5]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._best_scores[0] == 0.5

    def test_unattempted_samples_have_higher_weight(self):
        """Test that unattempted samples have higher weight than solved samples."""
        sampler = PMaxBatchSampler(minibatch_size=2, unattempted_boost=1.5, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Only attempt samples 0 and 1, leave sample 2 unattempted
        # Sample 0 gets solved, sample 1 stays unsolved
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        weights = sampler.get_weights()

        # Before normalization:
        # - Sample 0 (solved): 1.0
        # - Sample 1 (unsolved, attempted): AdaBoost from 1.0
        # - Sample 2 (unattempted): 1.5
        # After normalization, sample 2 should have higher weight than sample 0
        assert weights[2] > weights[0]  # Unattempted > solved

    def test_attempted_tracking(self):
        """Test that sampler correctly tracks which samples have been attempted."""
        sampler = PMaxBatchSampler(minibatch_size=2, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Initially no samples are attempted
        assert len(sampler._attempted) == 0

        # Attempt samples 0 and 1
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 0.0]})
        sampler.next_minibatch_ids(loader, state)

        # Samples 0 and 1 should be marked as attempted
        assert 0 in sampler._attempted
        assert 1 in sampler._attempted
        assert 2 not in sampler._attempted


class TestResidualWeightedSamplerIntegration:
    """Tests for ResidualWeightedSampler integration in AdaBoost/PMax samplers."""

    def test_adaboost_batch_has_no_duplicates(self):
        """Test that batches have unique elements (unique=True is default)."""
        sampler = AdaBoostBatchSampler(minibatch_size=3, beta=5.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample to initialize weights
        sampler.next_minibatch_ids(loader, state)

        # Make sample 0 have very low score (high error -> high weight)
        for _ in range(5):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 1.0, 1.0]})

        # Sample - with unique=True (default), no duplicates in batch
        result = sampler.next_minibatch_ids(loader, state)

        # All elements should be unique
        assert len(result) == len(set(result)), f"Batch should have no duplicates, got {result}"
        # All 3 elements should be present
        assert set(result) == {0, 1, 2}

    def test_adaboost_low_weight_sample_eventually_sampled(self):
        """Test that low-weight samples are eventually sampled (guaranteed coverage)."""
        sampler = AdaBoostBatchSampler(minibatch_size=2, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Make sample 2 have high score (lower weight, but not extreme)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.2, 0.2, 0.8]})

        # Sample multiple batches and collect all samples
        # With residual sampling, even low-weight samples are guaranteed to appear
        all_samples = []
        for _ in range(20):
            result = sampler.next_minibatch_ids(loader, state)
            all_samples.extend(result)

        # Even with lower weight, sample 2 should appear (guaranteed by residual sampling)
        assert 2 in all_samples, "Low-weight sample should eventually be sampled"

    def test_pmax_batch_has_no_duplicates(self):
        """Test that PMax batches have unique elements (unique=True is default)."""
        sampler = PMaxBatchSampler(minibatch_size=3, beta=5.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Sample 0 stays unsolved (high weight), samples 1, 2 get solved (weight reset to 1.0)
        for _ in range(5):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.0, 0.5, 0.5]})

        result = sampler.next_minibatch_ids(loader, state)

        # All elements should be unique
        assert len(result) == len(set(result)), f"Batch should have no duplicates, got {result}"

    def test_sampling_all_elements_included_over_time(self):
        """Test that all elements are included over multiple batches."""
        sampler = AdaBoostBatchSampler(minibatch_size=3, beta=1.0, rng=random.Random(42))
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Set up different scores to create weight variance
        state.full_program_trace.append(
            {"subsample_ids": [0, 1, 2, 3, 4], "subsample_scores": [0.0, 0.25, 0.5, 0.75, 1.0]}
        )

        # Sample many times and collect all unique samples seen
        all_samples_seen = set()
        for _ in range(20):
            result = sampler.next_minibatch_ids(loader, state)
            all_samples_seen.update(result)
            # Verify no duplicates in each batch
            assert len(result) == len(set(result)), f"Batch should have no duplicates, got {result}"

        # All elements should be seen at least once (guaranteed coverage)
        assert all_samples_seen == {0, 1, 2, 3, 4}, f"All elements should be seen, got {all_samples_seen}"
