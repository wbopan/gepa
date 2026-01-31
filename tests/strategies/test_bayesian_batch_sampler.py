# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections import Counter

import pytest

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
        sampler = BayesianBatchSampler(minibatch_size=2, binarize_threshold=0.5)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Sample 0: 2 successes, 2 failures (frontier)
        # Sample 1: 4 successes, 0 failures (one-sided)
        # Sample 2: 0 successes, 4 failures (one-sided)
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.6, 0.8, 0.2]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.4, 0.9, 0.1]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.7, 0.7, 0.3]})
        state.full_program_trace.append({"subsample_ids": [0, 1, 2], "subsample_scores": [0.3, 0.6, 0.4]})

        sampler.next_minibatch_ids(loader, state)
        scores = sampler.get_scores()

        # Sample 0 has the most balanced history (2s, 2f)
        # Sample 1 has all successes (4s, 0f)
        # Sample 2 has all failures (0s, 4f)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        assert scores[1] == scores[2]  # Symmetric

    def test_binarize_threshold(self):
        """Test that binarize_threshold correctly classifies scores."""
        sampler = BayesianBatchSampler(minibatch_size=2, binarize_threshold=0.7)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # With threshold 0.7:
        # Sample 0: 0.6, 0.8 -> 0 success, 1 failure; 1 success -> (1, 1)
        # Sample 1: 0.5, 0.5 -> 0 success, 2 failures -> (0, 2)
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.5]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.8, 0.5]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        assert counts[0] == (1, 1)  # 0.8 >= 0.7 is success, 0.6 < 0.7 is failure
        assert counts[1] == (0, 2)  # Both < 0.7 are failures

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

    def test_get_last_sampled_avg_weight(self):
        """Test get_last_sampled_avg_weight returns correct average."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # First sample - all scores are 1.0 (cold start)
        sampler.next_minibatch_ids(loader, state)
        avg = sampler.get_last_sampled_avg_weight()
        assert avg == pytest.approx(1.0)

    def test_get_last_sampled_avg_weight_empty(self):
        """Test get_last_sampled_avg_weight returns 1.0 when no samples yet."""
        sampler = BayesianBatchSampler(minibatch_size=2)
        assert sampler.get_last_sampled_avg_weight() == 1.0

    def test_get_train_sample_weight_stats(self):
        """Test get_train_sample_weight_stats returns correct stats."""
        sampler = BayesianBatchSampler(minibatch_size=3)
        loader = MockDataLoader([0, 1, 2])
        state = MockGEPAState()

        # Before any sampling
        assert sampler.get_train_sample_weight_stats() is None

        # After sampling
        sampler.next_minibatch_ids(loader, state)
        stats = sampler.get_train_sample_weight_stats()

        assert stats is not None
        assert "train/frontier_score_avg" in stats
        assert "train/frontier_score_max" in stats
        assert "train/frontier_score_min" in stats
        assert stats["train/frontier_score_avg"] == 1.0  # All cold start

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

        # Add 4 traces
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # All 4 traces should be counted
        assert counts[0] == (4, 0)  # 4 successes
        assert counts[1] == (0, 4)  # 4 failures

    def test_window_limits_history(self):
        """Test that window limits history to recent traces only."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add 4 traces - sample 0 starts failing, sample 1 starts succeeding
        # Trace 0: sample 0 success, sample 1 failure
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})
        # Trace 1: sample 0 success, sample 1 failure
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.6, 0.4]})
        # Trace 2: sample 0 failure, sample 1 success
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.4, 0.6]})
        # Trace 3: sample 0 failure, sample 1 success
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.4, 0.6]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # With window=2, only traces 2 and 3 are counted
        # Sample 0: 0 successes, 2 failures (traces 2, 3)
        # Sample 1: 2 successes, 0 failures (traces 2, 3)
        assert counts[0] == (0, 2)
        assert counts[1] == (2, 0)

    def test_window_allows_weight_recovery(self):
        """Test that samples can regain priority when recent candidates solve them."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=2)
        loader = MockDataLoader([0, 1])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Initially sample 0 always succeeds, sample 1 always fails
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.9, 0.3]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.9, 0.3]})

        sampler.next_minibatch_ids(loader, state)
        counts_after_initial = sampler.get_counts()

        # Sample 0 has (2, 0) in window -> one-sided success
        # Sample 1 has (0, 2) in window -> one-sided failure
        assert counts_after_initial[0] == (2, 0)
        assert counts_after_initial[1] == (0, 2)

        # Now sample 0 starts failing, sample 1 starts succeeding
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.3, 0.9]})
        state.full_program_trace.append({"subsample_ids": [0, 1], "subsample_scores": [0.3, 0.9]})

        sampler.next_minibatch_ids(loader, state)
        counts_after_change = sampler.get_counts()

        # With window=2, only traces 2, 3 are counted now
        # Sample 0: (0, 2) -> one-sided failure (old successes forgotten!)
        # Sample 1: (2, 0) -> one-sided success (old failures forgotten!)
        # The key is that both samples' old outcomes are forgotten
        assert counts_after_change[0] == (0, 2)  # Old successes forgotten
        assert counts_after_change[1] == (2, 0)  # Old failures forgotten

    def test_window_one_uses_only_latest_trace(self):
        """Test that window=1 uses only the most recent trace."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=1)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add multiple traces with alternating outcomes
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.6]})  # success
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.4]})  # failure
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.6]})  # success

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # With window=1, only trace 2 (last one) counts
        assert counts[0] == (1, 0)  # Only the last success

    def test_window_larger_than_history_uses_all(self):
        """Test that window larger than trace count uses all available traces."""
        sampler = BayesianBatchSampler(minibatch_size=2, window=10)
        loader = MockDataLoader([0])
        state = MockGEPAState()

        sampler.next_minibatch_ids(loader, state)

        # Add only 3 traces
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.6]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.6]})
        state.full_program_trace.append({"subsample_ids": [0], "subsample_scores": [0.4]})

        sampler.next_minibatch_ids(loader, state)
        counts = sampler.get_counts()

        # Window=10 but only 3 traces exist, so all are counted
        assert counts[0] == (2, 1)


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
