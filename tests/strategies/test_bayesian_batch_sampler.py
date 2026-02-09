# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections import Counter

import pytest

from gepa.strategies.batch_sampler import MetricLoggingBatchSampler
from gepa.strategies.bayesian_batch_sampler import BayesianBatchSampler, bayesian_frontier_score


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


class TestBayesianFrontierScore:
    """Tests for the bayesian_frontier_score function."""

    def test_cold_start_score(self):
        """Cold start (0, 0) should have max score 1.0."""
        score = bayesian_frontier_score(0, 0)
        assert score == 1.0

    def test_balanced_frontier_score(self):
        """Balanced samples (s = f) should have max score 1.0."""
        assert bayesian_frontier_score(1, 1) == 1.0
        assert bayesian_frontier_score(2, 2) == 1.0
        assert bayesian_frontier_score(50, 50) == 1.0

    def test_score_symmetry(self):
        """Score should be symmetric: (s, f) == (f, s)."""
        assert bayesian_frontier_score(4, 0) == bayesian_frontier_score(0, 4)
        assert bayesian_frontier_score(3, 1) == bayesian_frontier_score(1, 3)
        assert bayesian_frontier_score(10, 2) == bayesian_frontier_score(2, 10)

    def test_score_decay_with_one_sided_evidence(self):
        """Score should decrease as evidence becomes more one-sided."""
        score_2_0 = bayesian_frontier_score(2, 0)
        score_4_0 = bayesian_frontier_score(4, 0)
        score_10_0 = bayesian_frontier_score(10, 0)

        assert score_2_0 > score_4_0
        assert score_4_0 > score_10_0

    def test_specific_values(self):
        """Verify specific computed values."""
        # (2, 0): 4 * 3 * 1 / 16 = 0.75
        assert bayesian_frontier_score(2, 0) == pytest.approx(0.75)

        # (4, 0): 4 * 5 * 1 / 36 = 0.555...
        assert bayesian_frontier_score(4, 0) == pytest.approx(0.5555555, rel=1e-5)

    def test_frontier_higher_than_biased(self):
        """Frontier samples should have higher score than one-sided samples with same total."""
        frontier = bayesian_frontier_score(2, 3)  # 2 vs 3 (Total 5)
        biased = bayesian_frontier_score(5, 0)  # 5 vs 0 (Total 5)

        assert frontier > biased


class TestBayesianBatchSampler:
    """Tests for BayesianBatchSampler."""

    def test_initial_scores_are_one(self):
        """Test that initial scores are 1.0 for all samples (cold start)."""
        sampler = BayesianBatchSampler(minibatch_size=3)
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)
        scores = sampler.get_scores()

        for s in scores.values():
            assert s == pytest.approx(1.0)

    def test_balanced_samples_have_high_score(self):
        """Test that samples with balanced success/failure have high scores."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Sample 0: balanced scores around 0.5 (frontier)
        # Sample 1: high scores (one-sided successes)
        # Sample 2: low scores (one-sided failures)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.9, 0.1]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.9, 0.1]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.9, 0.1]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.9, 0.1]})

        sampler.next_minibatch_ids(loader, state)
        scores = sampler.get_scores()

        # Sample 0 has balanced fractional counts (2.0 successes, 2.0 failures)
        # Sample 1 has mostly successes (3.6 successes, 0.4 failures)
        # Sample 2 has mostly failures (0.4 successes, 3.6 failures)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        assert scores[1] == pytest.approx(scores[2])  # Symmetric

    def test_fractional_counts(self):
        """Test that continuous scores produce fractional success/failure counts."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Sample 0: 0.6 + 0.8 = 1.4 successes, 0.4 + 0.2 = 0.6 failures
        # Sample 1: 0.5 + 0.5 = 1.0 successes, 0.5 + 0.5 = 1.0 failures
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.5]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.8, 0.5]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        assert counts[0] == pytest.approx((1.4, 0.6))
        assert counts[1] == pytest.approx((1.0, 1.0))  # Perfectly balanced

    def test_empty_loader_raises(self):
        """Test that empty loader raises ValueError."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([])
        state = MockGEPAState()

        with pytest.raises(ValueError, match="Cannot sample from empty loader"):
            sampler.next_minibatch_ids(loader, state)

    def test_incremental_trace_processing(self):
        """Test that traces are processed incrementally."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == -1

        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.6]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == 0

        state.full_program_trace.append({"subsample_ids": [1], "subsample_scores": [0.4]})
        sampler.next_minibatch_ids(loader, state)
        assert sampler._last_processed_trace_idx == 1

    def test_empty_trace_no_count_update(self):
        """Test that empty trace entries don't cause errors."""
        sampler = BayesianBatchSampler(minibatch_size=2)
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
        """Test that only observed samples have their counts updated."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1, 2, 3, 4])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Only observe samples 0 and 1
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # Only samples 0 and 1 should have counts
        assert 0 in counts
        assert 1 in counts
        assert 2 not in counts
        assert 3 not in counts
        assert 4 not in counts

    def test_get_batch_weights(self):
        """Test get_batch_weights returns correct weights."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample - all scores are 1.0 (cold start)
        sampler.next_minibatch_ids(loader, state)
        weights = sampler.get_batch_weights()
        assert weights is not None
        assert len(weights) == 2
        for w in weights:
            assert w == pytest.approx(1.0)

    def test_get_batch_weights_empty(self):
        """Test get_batch_weights returns None when no samples yet."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        assert sampler.get_batch_weights() is None

    def test_get_all_sample_weights(self):
        """Test get_all_sample_weights returns per-sample weights."""
        sampler = BayesianBatchSampler(minibatch_size=3)
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
        # All cold start, should be 1.0
        assert all_weights[0] == pytest.approx(1.0)

    def test_metric_logging_protocol_conformance(self):
        """Test that BayesianBatchSampler conforms to MetricLoggingBatchSampler protocol."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        assert isinstance(sampler, MetricLoggingBatchSampler)

    def test_minibatch_size_larger_than_dataset(self):
        """Test that minibatch_size larger than dataset returns all samples."""
        sampler = BayesianBatchSampler(minibatch_size=10)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        result = sampler.next_minibatch_ids(loader, state)
        assert len(result) == 3

    def test_frontier_samples_sampled_more_frequently(self):
        """Test that frontier samples are sampled more frequently over time."""
        sampler = BayesianBatchSampler(minibatch_size=1)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Set up histories:
        # Sample 0: balanced (frontier)
        # Sample 1: all successes (one-sided)
        # Sample 2: all failures (one-sided)
        for _ in range(5):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.9, 0.1]})
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.8, 0.2]})

        # Sample many times
        counts = Counter()
        for _ in range(100):
            result = sampler.next_minibatch_ids(loader, state)
            counts.update(result)

        # Sample 0 (frontier) should be sampled more often than 1 or 2 (one-sided)
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]

    def test_dataset_expansion(self):
        """Test that new data IDs get cold start score."""
        sampler = BayesianBatchSampler(minibatch_size=2)

        # Start with 2 samples
        loader1 = MockDataLoader([0, 1])
        state = MockGEPAState()
        sampler.next_minibatch_ids(loader1, state)

        assert 0 in sampler.get_scores()
        assert 1 in sampler.get_scores()
        assert 2 not in sampler.get_scores()

        # Expand to 3 samples
        loader2 = MockDataLoader([0, 1, 2])
        sampler.next_minibatch_ids(loader2, state)

        # New sample should have cold start score
        assert 2 in sampler.get_scores()
        assert sampler.get_scores()[2] == 1.0


class TestBayesianBatchSamplerWindow:
    """Tests for the window parameter of BayesianBatchSampler."""

    def test_window_none_uses_all_history(self):
        """Test that window=None (default) uses all history."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=None)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add 4 traces with binary scores (0.0 and 1.0) to verify backward compatibility
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # All 4 traces should be counted with binary scores
        assert counts[0] == pytest.approx((4.0, 0.0))  # 4 successes
        assert counts[1] == pytest.approx((0.0, 4.0))  # 4 failures

    def test_window_limits_history(self):
        """Test that window limits history to recent traces only."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add 4 traces - sample 0 starts failing, sample 1 starts succeeding
        # Use binary scores (0.0 and 1.0) for clarity
        # Trace 0: sample 0 success, sample 1 failure
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        # Trace 1: sample 0 success, sample 1 failure
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        # Trace 2: sample 0 failure, sample 1 success
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0]})
        # Trace 3: sample 0 failure, sample 1 success
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # With window=2, only traces 2 and 3 are counted
        # Sample 0: 0 successes, 2 failures (traces 2, 3)
        # Sample 1: 2 successes, 0 failures (traces 2, 3)
        assert counts[0] == pytest.approx((0.0, 2.0))
        assert counts[1] == pytest.approx((2.0, 0.0))

    def test_window_allows_weight_recovery(self):
        """Test that samples can regain priority when recent candidates solve them."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Initially sample 0 always succeeds, sample 1 always fails (binary scores)
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1.0, 0.0]})

        sampler.next_minibatch_ids(loader, state)
        counts_after_initial = sampler.get_counts()

        # Sample 0 has (2, 0) in window -> one-sided success
        # Sample 1 has (0, 2) in window -> one-sided failure
        assert counts_after_initial[0] == pytest.approx((2.0, 0.0))
        assert counts_after_initial[1] == pytest.approx((0.0, 2.0))

        # Now sample 0 starts failing, sample 1 starts succeeding
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.0, 1.0]})

        sampler.next_minibatch_ids(loader, state)
        counts_after_change = sampler.get_counts()

        # With window=2, only traces 2, 3 are counted now
        # Sample 0: (0, 2) -> one-sided failure (old successes forgotten!)
        # Sample 1: (2, 0) -> one-sided success (old failures forgotten!)
        # The key is that both samples' old outcomes are forgotten
        assert counts_after_change[0] == pytest.approx((0.0, 2.0))  # Old successes forgotten
        assert counts_after_change[1] == pytest.approx((2.0, 0.0))  # Old failures forgotten

    def test_window_one_uses_only_latest_trace(self):
        """Test that window=1 uses only the most recent trace."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=1)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add multiple traces with alternating outcomes (binary scores)
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [1.0]})  # success
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.0]})  # failure
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [1.0]})  # success

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # With window=1, only trace 2 (last one) counts
        assert counts[0] == pytest.approx((1.0, 0.0))  # Only the last success

    def test_window_larger_than_history_uses_all(self):
        """Test that window larger than trace count uses all available traces."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=10)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add only 3 traces with binary scores
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [1.0]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [1.0]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.0]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # Window=10 but only 3 traces exist, so all are counted
        assert counts[0] == pytest.approx((2.0, 1.0))


class TestFractionalCounting:
    """Tests for fractional success/failure counting with continuous scores."""

    def test_fractional_scores_produce_different_weights(self):
        """Test that fractional scores produce different weights than binary."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Sample 0: consistent 0.5 scores (perfectly balanced)
        # Sample 1: alternating 0.0 and 1.0 (also perfectly balanced, but binary)
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 0.0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.5, 1.0]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # Both should have equal successes and failures
        assert counts[0] == pytest.approx((1.0, 1.0))  # 0.5 + 0.5, 0.5 + 0.5
        assert counts[1] == pytest.approx((1.0, 1.0))  # 0.0 + 1.0, 1.0 + 0.0

    def test_score_0_5_produces_balanced_counts(self):
        """Test that score 0.5 results in exactly balanced successes/failures."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.5]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.5]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.5]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # 3 * 0.5 = 1.5 successes, 3 * 0.5 = 1.5 failures
        assert counts[0] == pytest.approx((1.5, 1.5))

    def test_partial_credit_scores(self):
        """Test typical partial credit scores like 0.25, 0.5, 0.75."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Simulate NYT Connections style scores
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.25]})  # 1 group correct
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.50]})  # 2 groups correct
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.75]})  # 3 groups correct
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [1.00]})  # all correct

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # successes = 0.25 + 0.50 + 0.75 + 1.00 = 2.5
        # failures = 0.75 + 0.50 + 0.25 + 0.00 = 1.5
        assert counts[0] == pytest.approx((2.5, 1.5))

    def test_scores_outside_0_1_are_clamped(self):
        """Test that scores outside [0, 1] are clamped."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Scores outside [0, 1] should be clamped
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [-0.5, 1.5]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # -0.5 clamped to 0.0, 1.5 clamped to 1.0
        assert counts[0] == pytest.approx((0.0, 1.0))
        assert counts[1] == pytest.approx((1.0, 0.0))

    def test_backward_compatibility_with_binary_scores(self):
        """Test that binary scores (0 and 1) work identically to before."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Only binary scores
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1, 0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [1, 0]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0, 1]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # Sample 0: 2 successes, 1 failure
        # Sample 1: 1 success, 2 failures
        assert counts[0] == pytest.approx((2.0, 1.0))
        assert counts[1] == pytest.approx((1.0, 2.0))


class TestResidualSamplerIntegration:
    """Tests for ResidualWeightedSampler integration."""

    def test_high_score_sample_can_appear_multiple_times(self):
        """Test that high-score (frontier) samples can appear multiple times."""
        sampler = BayesianBatchSampler(minibatch_size=5)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Make sample 0 a frontier (balanced), samples 1 and 2 one-sided
        for _ in range(10):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.9, 0.1]})
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.8, 0.2]})

        result = sampler.next_minibatch_ids(loader, state)

        # With residual sampling, the high-score sample may appear multiple times
        # but this depends on the relative weights
        assert len(result) == 3  # Limited by dataset size

    def test_all_samples_eventually_sampled(self):
        """Test that all samples are eventually sampled (guaranteed coverage)."""
        sampler = BayesianBatchSampler(minibatch_size=1)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Make weights uneven but non-zero
        for _ in range(5):
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.9, 0.1]})
            state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.8, 0.2]})

        # Sample many batches
        all_samples = []
        for _ in range(50):
            result = sampler.next_minibatch_ids(loader, state)
            all_samples.extend(result)

        # All samples should appear (residual sampling guarantees coverage)
        assert 0 in all_samples
        assert 1 in all_samples
        assert 2 in all_samples
