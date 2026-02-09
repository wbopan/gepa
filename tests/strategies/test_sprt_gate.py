# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.sprt_gate import SPRTDecision, SPRTGate


class TestSPRTGateBasics:
    """Basic tests for SPRTGate initialization and thresholds."""

    def test_default_thresholds(self):
        """Test default SPRT thresholds with α=β=0.2, p0=0.4, p1=0.6."""
        gate = SPRTGate()

        # threshold_accept = log((1-0.2)/0.2) = log(4) ≈ 1.386
        assert gate.threshold_accept == pytest.approx(1.386, rel=0.01)

        # threshold_reject = log(0.2/(1-0.2)) = log(0.25) ≈ -1.386
        assert gate.threshold_reject == pytest.approx(-1.386, rel=0.01)

        # llr_win = log(0.6/0.4) ≈ 0.405
        assert gate.llr_win == pytest.approx(0.405, rel=0.01)

        # llr_loss = log(0.4/0.6) ≈ -0.405
        assert gate.llr_loss == pytest.approx(-0.405, rel=0.01)

    def test_custom_parameters(self):
        """Test SPRT with custom parameters."""
        gate = SPRTGate(alpha=0.1, beta=0.1, p0=0.3, p1=0.7)

        # Thresholds should be different with different alpha/beta
        assert gate.threshold_accept != pytest.approx(1.386, rel=0.01)
        assert gate.threshold_reject != pytest.approx(-1.386, rel=0.01)


class TestSPRTGateEvaluateInitialBatch:
    """Tests for evaluate_initial_batch method."""

    def test_early_accept_4_consecutive_wins(self):
        """Test 1: Early accept with 4 consecutive wins."""
        gate = SPRTGate()

        parent_scores = [0, 0, 0, 0, 0]
        child_scores = [1, 1, 1, 1, 0]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.ACCEPT
        assert result.samples_evaluated == 4  # Decided at sample 4, not 5
        assert result.wins == 4
        assert result.losses == 0
        assert result.ties == 0
        assert result.reason == "early_accept"
        assert result.llr >= gate.threshold_accept

    def test_early_reject_4_consecutive_losses(self):
        """Test 2: Early reject with 4 consecutive losses."""
        gate = SPRTGate()

        parent_scores = [1, 1, 1, 1, 0]
        child_scores = [0, 0, 0, 0, 0]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.REJECT
        assert result.samples_evaluated == 4
        assert result.wins == 0
        assert result.losses == 4
        assert result.ties == 0
        assert result.reason == "early_reject"
        assert result.llr <= gate.threshold_reject

    def test_all_ties_undecided(self):
        """Test 3: All ties results in undecided."""
        gate = SPRTGate()

        parent_scores = [1, 0, 1, 0, 1]
        child_scores = [1, 0, 1, 0, 1]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.UNDECIDED
        assert result.samples_evaluated == 5
        assert result.wins == 0
        assert result.losses == 0
        assert result.ties == 5
        assert result.llr == pytest.approx(0.0)
        assert result.reason == "undecided"

    def test_alternating_win_loss_undecided(self):
        """Test 4: Alternating W-L results in undecided."""
        gate = SPRTGate()

        parent_scores = [0, 1, 0, 1, 0]
        child_scores = [1, 0, 1, 0, 1]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.UNDECIDED
        assert result.samples_evaluated == 5
        assert result.wins == 3
        assert result.losses == 2
        # LLR = 3 * 0.405 + 2 * (-0.405) = 0.405
        assert result.llr == pytest.approx(0.405, rel=0.01)
        assert result.reason == "undecided"

    def test_3_wins_0_losses_still_undecided(self):
        """Test 5: 3 wins, 0 losses - still undecided (boundary)."""
        gate = SPRTGate()

        parent_scores = [0, 0, 0, 1, 1]
        child_scores = [1, 1, 1, 1, 1]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.UNDECIDED
        assert result.samples_evaluated == 5
        assert result.wins == 3
        assert result.losses == 0
        assert result.ties == 2
        # LLR = 3 * 0.405 = 1.215 < 1.386
        assert result.llr == pytest.approx(1.215, rel=0.01)
        assert result.llr < gate.threshold_accept
        assert result.reason == "undecided"

    def test_decision_order_matters(self):
        """Test 12: Order of wins/losses matters for threshold crossing."""
        gate = SPRTGate()

        parent_scores = [0, 0, 0, 1, 1]
        child_scores = [1, 1, 1, 0, 0]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        # W, W, W, L, L -> LLR after each: 0.405, 0.810, 1.215, 0.810, 0.405
        # Never crosses threshold, ends undecided
        assert result.decision == SPRTDecision.UNDECIDED
        assert result.samples_evaluated == 5
        assert result.wins == 3
        assert result.losses == 2
        assert result.llr == pytest.approx(0.405, rel=0.01)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched parent/child score lengths raise ValueError."""
        gate = SPRTGate()

        with pytest.raises(ValueError, match="same length"):
            gate.evaluate_initial_batch([0, 1, 2], [0, 1])


class TestSPRTGateContinuation:
    """Tests for continue_evaluation method."""

    def test_continuation_accept_after_more_samples(self):
        """Test 6: Continuation - accept after more samples."""
        gate = SPRTGate()

        # Initial batch: W=2, L=2, T=1 -> LLR = 0
        parent_scores_initial = [0, 1, 0, 1, 0.5]
        child_scores_initial = [1, 0, 1, 0, 0.5]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED
        assert result.llr == pytest.approx(0.0, abs=0.01)

        # Continue with child winning: 4 more wins needed to accept
        remaining_ids = [5, 6, 7, 8]

        def mock_fetch(ids):
            return [{"id": i} for i in ids]

        def mock_parent_eval(batch):
            return [0.0] * len(batch)

        def mock_child_eval(batch):
            return [1.0] * len(batch)

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        assert result.decision == SPRTDecision.ACCEPT
        assert result.reason == "early_accept"
        assert result.samples_evaluated == 9  # 5 initial + 4 continuation

    def test_continuation_reject_after_more_samples(self):
        """Test 7: Continuation - reject after more samples."""
        gate = SPRTGate()

        # Initial batch: W=2, L=2, T=1 -> LLR = 0
        parent_scores_initial = [0, 1, 0, 1, 0.5]
        child_scores_initial = [1, 0, 1, 0, 0.5]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED

        # Continue with child losing
        remaining_ids = [5, 6, 7, 8]

        def mock_fetch(ids):
            return [{"id": i} for i in ids]

        def mock_parent_eval(batch):
            return [1.0] * len(batch)

        def mock_child_eval(batch):
            return [0.0] * len(batch)

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        assert result.decision == SPRTDecision.REJECT
        assert result.reason == "early_reject"
        assert result.samples_evaluated == 9

    def test_max_samples_greedy_accept(self):
        """Test 8: Max samples exhausted - greedy accept (LLR > 0)."""
        gate = SPRTGate()

        # Initial 5: W=3, L=2 -> LLR = 0.405
        parent_scores_initial = [0, 1, 0, 1, 0]
        child_scores_initial = [1, 0, 1, 0, 1]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED
        assert result.llr == pytest.approx(0.405, rel=0.01)

        # Only 2 more samples available, both ties
        remaining_ids = [5, 6]

        def mock_fetch(ids):
            return [{"id": i} for i in ids]

        def mock_parent_eval(batch):
            return [0.5] * len(batch)

        def mock_child_eval(batch):
            return [0.5] * len(batch)

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        assert result.decision == SPRTDecision.ACCEPT
        assert result.reason == "max_samples"
        assert result.llr > 0
        assert result.samples_evaluated == 7

    def test_max_samples_greedy_reject(self):
        """Test 9: Max samples exhausted - greedy reject (LLR < 0)."""
        gate = SPRTGate()

        # Initial 5: W=2, L=3 -> LLR = -0.405
        parent_scores_initial = [1, 0, 1, 0, 1]
        child_scores_initial = [0, 1, 0, 1, 0]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED
        assert result.llr == pytest.approx(-0.405, rel=0.01)

        # Only 2 more samples, both ties
        remaining_ids = [5, 6]

        def mock_fetch(ids):
            return [{"id": i} for i in ids]

        def mock_parent_eval(batch):
            return [0.5] * len(batch)

        def mock_child_eval(batch):
            return [0.5] * len(batch)

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        assert result.decision == SPRTDecision.REJECT
        assert result.reason == "max_samples"
        assert result.llr < 0
        assert result.samples_evaluated == 7

    def test_max_samples_llr_exactly_zero(self):
        """Test 10: Max samples exhausted - LLR exactly 0 -> reject."""
        gate = SPRTGate()

        # Initial 5: W=2, L=2, T=1 -> LLR = 0
        parent_scores_initial = [0, 1, 0, 1, 0.5]
        child_scores_initial = [1, 0, 1, 0, 0.5]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED
        assert result.llr == pytest.approx(0.0, abs=0.01)

        # Only 2 more samples, both ties
        remaining_ids = [5, 6]

        def mock_fetch(ids):
            return [{"id": i} for i in ids]

        def mock_parent_eval(batch):
            return [0.5] * len(batch)

        def mock_child_eval(batch):
            return [0.5] * len(batch)

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        assert result.decision == SPRTDecision.REJECT
        assert result.reason == "max_samples"
        assert result.llr == pytest.approx(0.0, abs=0.01)

    def test_empty_remaining_samples(self):
        """Test 11: Empty remaining samples -> greedy decision."""
        gate = SPRTGate()

        # Initial 5: W=2, L=2, T=1 -> LLR = 0
        parent_scores_initial = [0, 1, 0, 1, 0.5]
        child_scores_initial = [1, 0, 1, 0, 0.5]

        result = gate.evaluate_initial_batch(parent_scores_initial, child_scores_initial)
        assert result.decision == SPRTDecision.UNDECIDED

        # No remaining samples
        remaining_ids: list = []

        def mock_fetch(ids):
            return []

        def mock_parent_eval(batch):
            return []

        def mock_child_eval(batch):
            return []

        result = gate.continue_evaluation(remaining_ids, mock_fetch, mock_parent_eval, mock_child_eval)

        # LLR = 0, should reject
        assert result.decision == SPRTDecision.REJECT
        assert result.reason == "max_samples"
        assert result.samples_evaluated == 5  # No additional


class TestSPRTGateReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears internal state."""
        gate = SPRTGate()

        # Run initial evaluation
        gate.evaluate_initial_batch([0, 0, 0], [1, 1, 1])
        assert gate._llr != 0
        assert gate._wins == 3

        # Reset
        gate.reset()

        assert gate._llr == 0.0
        assert gate._wins == 0
        assert gate._losses == 0
        assert gate._ties == 0
        assert gate._samples_evaluated == 0

    def test_evaluate_initial_batch_auto_resets(self):
        """Test that evaluate_initial_batch automatically resets state."""
        gate = SPRTGate()

        # First evaluation
        result1 = gate.evaluate_initial_batch([0, 0], [1, 1])
        assert result1.wins == 2

        # Second evaluation should start fresh
        result2 = gate.evaluate_initial_batch([1, 1], [0, 0])
        assert result2.wins == 0
        assert result2.losses == 2


class TestSPRTGateStatistics:
    """Tests for statistics tracking."""

    def test_get_additional_evals_count(self):
        """Test additional evals count calculation."""
        gate = SPRTGate()

        # Initial batch of 5
        gate.evaluate_initial_batch([0, 0, 0, 0, 0], [1, 0, 1, 0, 1])

        # 2 additional samples would need 4 evals (2 parent + 2 child)
        # But we need to simulate continuation to update _samples_evaluated
        gate._samples_evaluated = 7  # 5 initial + 2 continuation

        assert gate.get_additional_evals_count(5) == 4

    def test_win_loss_tie_tracking(self):
        """Test accurate tracking of wins, losses, and ties."""
        gate = SPRTGate()

        # 2 wins, 1 loss, 2 ties
        parent_scores = [0, 1, 0.5, 0, 0.5]
        child_scores = [1, 0, 0.5, 1, 0.5]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.wins == 2
        assert result.losses == 1
        assert result.ties == 2
        assert result.samples_evaluated == 5


class TestSPRTGateEdgeCases:
    """Edge case tests."""

    def test_single_sample_win(self):
        """Test with single sample that's a win (not enough to decide)."""
        gate = SPRTGate()

        result = gate.evaluate_initial_batch([0], [1])

        assert result.decision == SPRTDecision.UNDECIDED
        assert result.wins == 1
        assert result.llr == pytest.approx(0.405, rel=0.01)

    def test_large_batch_early_decision(self):
        """Test that large batch still decides early."""
        gate = SPRTGate()

        # 100 samples, first 4 are wins
        parent_scores = [0] * 100
        child_scores = [1] * 4 + [0] * 96

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        assert result.decision == SPRTDecision.ACCEPT
        assert result.samples_evaluated == 4
        assert result.reason == "early_accept"

    def test_float_score_comparison(self):
        """Test with float scores."""
        gate = SPRTGate()

        parent_scores = [0.3, 0.5, 0.7, 0.9, 0.1]
        child_scores = [0.4, 0.4, 0.8, 0.8, 0.2]

        result = gate.evaluate_initial_batch(parent_scores, child_scores)

        # Sample 1: 0.4 > 0.3 -> win
        # Sample 2: 0.4 < 0.5 -> loss
        # Sample 3: 0.8 > 0.7 -> win
        # Sample 4: 0.8 < 0.9 -> loss
        # Sample 5: 0.2 > 0.1 -> win
        assert result.wins == 3
        assert result.losses == 2
        assert result.ties == 0
