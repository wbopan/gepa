# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections import Counter

import pytest

from gepa.strategies.residual_weighted_sampler import ResidualWeightedSampler


class TestResidualWeightedSampler:
    """Tests for ResidualWeightedSampler."""

    def test_uniform_weights_each_appears_once_per_round(self):
        """With weight=1.0 for all, each element appears exactly once per round."""
        sampler = ResidualWeightedSampler(5)
        sampler.update_weights([1.0] * 5)

        # Sample exactly one round (5 elements)
        result = sampler.sample(5)
        assert sorted(result) == [0, 1, 2, 3, 4]

        # Sample another round
        result = sampler.sample(5)
        assert sorted(result) == [0, 1, 2, 3, 4]

    def test_half_weight_appears_once_every_two_rounds(self):
        """With weight=0.5, element appears once every 2 rounds."""
        sampler = ResidualWeightedSampler(1)
        sampler.update_weights([0.5])

        # First round: accumulator 0 + 0.5 = 0.5, count=0
        # Second round: accumulator 0.5 + 0.5 = 1.0, count=1
        samples_round_1_and_2 = sampler.sample(1)
        assert samples_round_1_and_2 == [0]

        # Third and fourth round: same pattern
        samples_round_3_and_4 = sampler.sample(1)
        assert samples_round_3_and_4 == [0]

    def test_one_and_half_weight_appears_three_times_in_two_rounds(self):
        """With weight=1.5, element appears 3 times in 2 rounds (unique=False)."""
        sampler = ResidualWeightedSampler(1)
        sampler.update_weights([1.5])

        # Round 1: acc = 0 + 1.5 = 1.5, count=1, acc=0.5
        # Round 2: acc = 0.5 + 1.5 = 2.0, count=2, acc=0
        result = sampler.sample(3, unique=False)
        assert result == [0, 0, 0]

    def test_weight_two_appears_twice_per_round(self):
        """With weight=2.0, element appears twice per round (unique=False)."""
        sampler = ResidualWeightedSampler(1)
        sampler.update_weights([2.0])

        result = sampler.sample(4, unique=False)  # 2 rounds
        assert result == [0, 0, 0, 0]

    def test_weight_gte_one_always_sampled_every_round(self):
        """Elements with weight >= 1 must appear at least once per round (unique=False)."""
        sampler = ResidualWeightedSampler(5)
        sampler.update_weights([1.0, 1.2, 1.5, 2.0, 3.0])

        # sum(weights) = 8.7, so each "round" produces ~8.7 samples
        # Over 10 rounds, each element with w>=1 should appear at least 10 times
        total_samples = 87  # ~10 rounds
        result = sampler.sample(total_samples, unique=False)

        counts = Counter(result)
        # Each element should appear at least 10 times (once per round minimum)
        for i in range(5):
            assert counts[i] >= 10, f"Element {i} appeared only {counts[i]} times"

    def test_mixed_weights_distribution(self):
        """Verify statistical correctness with mixed weights (unique=False)."""
        sampler = ResidualWeightedSampler(5)
        weights = [2.0, 0.0, 1.5, 0.5, 1.0]  # sum = 5.0
        sampler.update_weights(weights)

        # Sample 500 elements (100 rounds)
        total_samples = 500
        result = sampler.sample(total_samples, unique=False)

        counts = Counter(result)
        rounds = total_samples / sum(weights)  # 100 rounds

        # Verify counts match expected values closely
        for i, w in enumerate(weights):
            expected = w * rounds
            actual = counts[i]
            # Allow small error due to rounding
            assert abs(actual - expected) <= 1, f"Element {i}: expected {expected}, got {actual}"

    def test_zero_weight_never_sampled(self):
        """Elements with weight=0 should never appear (unique=False)."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([1.0, 0.0, 1.0])

        result = sampler.sample(100, unique=False)
        assert 1 not in result

    def test_low_weight_eventually_sampled(self):
        """Low weight elements must eventually be sampled (guaranteed coverage, unique=False)."""
        sampler = ResidualWeightedSampler(2)
        sampler.update_weights([1.0, 0.1])

        # Weight 0.1 means it appears once every 10 rounds
        # sum(weights) = 1.1, so 25 samples ≈ 22.7 rounds
        # Element 1 should appear at least twice (once per 10 rounds)
        result = sampler.sample(25, unique=False)

        counts = Counter(result)
        assert counts[1] >= 2, f"Low weight element should appear at least 2 times, got {counts[1]}"

    def test_dynamic_weight_update_preserves_accumulator(self):
        """Updating weights preserves accumulator state for smooth transitions."""
        sampler = ResidualWeightedSampler(1)
        sampler.update_weights([0.3])

        # Accumulate some progress: 3 rounds = 0.9 accumulated
        sampler.sample(0)  # Just advance cursor without consuming
        # Manually simulate: we need to advance the sampler
        # Actually, sample(0) returns empty and doesn't advance
        # Let's do 3 rounds worth of advancement
        for _ in range(3):
            sampler._accumulators[0] += 0.3

        # Accumulator should be ~0.9 now
        assert sampler._accumulators[0] == pytest.approx(0.9, rel=0.01)

        # Change weight to 0.2
        sampler.update_weights([0.2])

        # Accumulator is still 0.9, next round adds 0.2 -> 1.1, output 1
        sampler._accumulators[0] += 0.2
        assert sampler._accumulators[0] == pytest.approx(1.1, rel=0.01)

    def test_reset_clears_state(self):
        """Reset should clear accumulators, cursor, and buffer but keep weights."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([1.0, 2.0, 0.5])

        # Advance state (use unique=False to sample more than n)
        sampler.sample(5, unique=False)

        # Reset
        sampler.reset()

        assert sampler._accumulators == [0.0, 0.0, 0.0]
        assert sampler._cursor == 0
        assert len(sampler._buffer) == 0
        # Weights should be preserved
        assert sampler._weights == [1.0, 2.0, 0.5]

    def test_buffer_preserves_excess(self):
        """Excess samples should be buffered for the next sample() call (unique=False)."""
        sampler = ResidualWeightedSampler(1)
        sampler.update_weights([3.0])

        # First sample(1): produces 3, returns 1, buffers 2
        result1 = sampler.sample(1, unique=False)
        assert result1 == [0]
        assert len(sampler._buffer) == 2

        # Next sample(1): returns from buffer
        result2 = sampler.sample(1, unique=False)
        assert result2 == [0]
        assert len(sampler._buffer) == 1

    def test_sample_more_than_one_round(self):
        """Sampling more than sum(weights) should span multiple rounds (unique=False)."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([1.0, 1.0, 1.0])  # sum = 3

        # Sample 9 elements = 3 complete rounds
        result = sampler.sample(9, unique=False)

        counts = Counter(result)
        assert counts[0] == 3
        assert counts[1] == 3
        assert counts[2] == 3

    def test_invalid_weights_length_raises(self):
        """Updating with wrong length weights should raise ValueError."""
        sampler = ResidualWeightedSampler(3)

        with pytest.raises(ValueError, match="Weights length must be 3"):
            sampler.update_weights([1.0, 2.0])

    def test_order_is_round_robin_with_weights(self):
        """Elements should appear in round-robin order, weighted by their weights (unique=False)."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([1.0, 0.5, 2.0])  # sum = 3.5

        # First round: 0 (1x), 2 (2x), then next round starts
        # Pattern: 0 appears at positions aligned with cursor
        result = sampler.sample(7, unique=False)  # 2 complete rounds

        # Verify pattern:
        # Round 1: cursor 0 -> acc=1.0, emit 0
        #          cursor 1 -> acc=0.5, emit nothing
        #          cursor 2 -> acc=2.0, emit 2,2
        # Round 2: cursor 0 -> acc=1.0, emit 0
        #          cursor 1 -> acc=1.0, emit 1
        #          cursor 2 -> acc=2.0, emit 2,2
        assert result == [0, 2, 2, 0, 1, 2, 2]

    def test_all_zero_weights_produces_nothing(self):
        """With all zero weights, sampling loops forever. Test empty sample."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([0.0, 0.0, 0.0])

        # sample(0) should return empty without infinite loop
        result = sampler.sample(0)
        assert result == []

        # Note: sample(k) where k > 0 with all-zero weights will loop forever.
        # This is expected behavior - caller should ensure sum(weights) > 0.


class TestResidualWeightedSamplerUnique:
    """Tests for unique=True behavior (default)."""

    def test_unique_no_duplicates_in_single_batch(self):
        """With unique=True (default), high weight item appears only once per batch."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([5.0, 1.0, 1.0])

        # Even though weight is 5.0, element 0 should appear only once
        result = sampler.sample(3)
        assert len(result) == 3
        assert len(set(result)) == 3  # All unique
        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_unique_deferred_sampling_accumulates(self):
        """Excess weight is preserved in accumulator for next batch."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([5.0, 1.0, 1.0])

        # First batch: element 0 appears once, but accumulates excess
        result1 = sampler.sample(3)
        assert sorted(result1) == [0, 1, 2]

        # Check accumulator state: element 0 should have accumulated weight
        # After first batch: acc[0] should be ~4.0 (started with 5.0, emitted once, subtracted 1.0)
        # But due to round-robin, it depends on traversal order
        assert sampler._accumulators[0] > 0  # Some accumulated weight

    def test_unique_preserves_accumulator_across_batches(self):
        """High weight items accumulate credit across batches (unique=True)."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([5.0, 1.0, 1.0])

        # Sample 3 batches of size 1
        sampler.sample(1)  # Cursor at 0, selects 0, acc[0] = 4.0, cursor -> 1
        sampler.sample(1)  # Cursor at 1, selects 1, acc[1] = 0.0, cursor -> 2
        sampler.sample(1)  # Cursor at 2, selects 2, acc[2] = 0.0, cursor -> 0

        # Element 0 should have accumulated credit
        assert sampler._accumulators[0] == 4.0
        # Elements 1 and 2 should have 0 credit
        assert sampler._accumulators[1] == 0.0
        assert sampler._accumulators[2] == 0.0

    def test_unique_with_k_greater_than_n_raises(self):
        """unique=True with k > n should raise ValueError."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="Cannot sample 5 unique elements"):
            sampler.sample(5)

    def test_unique_with_k_greater_than_nonzero_raises(self):
        """unique=True with k > non-zero weight items should raise ValueError."""
        sampler = ResidualWeightedSampler(5)
        sampler.update_weights([1.0, 0.0, 1.0, 0.0, 0.0])  # Only 2 non-zero

        with pytest.raises(ValueError, match="only 2 elements have non-zero weight"):
            sampler.sample(3)

    def test_unique_false_allows_duplicates(self):
        """Explicit test that unique=False allows duplicates."""
        sampler = ResidualWeightedSampler(3)
        sampler.update_weights([3.0, 0.0, 0.0])  # Only element 0 has weight

        result = sampler.sample(3, unique=False)
        assert result == [0, 0, 0]

    def test_unique_all_elements_eventually_selected(self):
        """With unique=True, all elements with non-zero weight are eventually selected."""
        sampler = ResidualWeightedSampler(5)
        sampler.update_weights([3.0, 1.0, 1.0, 0.5, 0.5])

        # Sample many batches of size 3
        counts = Counter()
        for _ in range(20):
            result = sampler.sample(3)
            counts.update(result)
            # Verify no duplicates in batch
            assert len(set(result)) == 3

        # All elements should have been selected at least once
        for i in range(5):
            assert counts[i] > 0, f"Element {i} was never selected"

    def test_unique_default_is_true(self):
        """Verify that unique defaults to True."""
        sampler = ResidualWeightedSampler(2)
        sampler.update_weights([2.0, 1.0])

        # With unique=True (default), no duplicates in batch of 2
        result = sampler.sample(2)
        assert len(set(result)) == 2
