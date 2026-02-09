# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId


class SPRTDecision(Enum):
    """SPRT decision outcomes."""

    ACCEPT = "accept"
    REJECT = "reject"
    UNDECIDED = "undecided"


@dataclass
class SPRTResult:
    """Result from SPRT evaluation."""

    decision: SPRTDecision
    llr: float
    samples_evaluated: int
    wins: int
    losses: int
    ties: int
    reason: str  # "early_accept", "early_reject", "max_samples", "undecided"


class SPRTGate:
    """
    Sequential Probability Ratio Test (SPRT) gate for candidate selection.

    Uses paired comparisons between parent and child scores to make early
    accept/reject decisions. A "win" is when child > parent, "loss" is when
    child < parent, and "tie" is when child == parent.

    Default parameters (α=β=0.2, p0=0.4, p1=0.6):
    - threshold_accept = log((1-β)/α) = +1.386
    - threshold_reject = log(β/(1-α)) = -1.386
    - llr_win = log(p1/p0) = +0.405
    - llr_loss = log((1-p1)/(1-p0)) = -0.405

    Need 4 net wins (wins - losses >= 4) to accept, 4 net losses to reject.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.2,
        p0: float = 0.4,
        p1: float = 0.6,
    ):
        """
        Initialize SPRT gate with error rates and effect size parameters.

        Args:
            alpha: Type I error rate (false positive). Default 0.2.
            beta: Type II error rate (false negative). Default 0.2.
            p0: Null hypothesis probability (child win rate under H0). Default 0.4.
            p1: Alternative hypothesis probability (child win rate under H1). Default 0.6.
        """
        self.alpha = alpha
        self.beta = beta
        self.p0 = p0
        self.p1 = p1

        # Compute thresholds
        self.threshold_accept = math.log((1 - beta) / alpha)
        self.threshold_reject = math.log(beta / (1 - alpha))

        # Compute LLR increments
        self.llr_win = math.log(p1 / p0)
        self.llr_loss = math.log((1 - p1) / (1 - p0))

        # State for continuation
        self._llr = 0.0
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._samples_evaluated = 0

    def reset(self) -> None:
        """Reset internal state for a new comparison."""
        self._llr = 0.0
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._samples_evaluated = 0

    def _update_llr(self, parent_score: float, child_score: float) -> SPRTDecision:
        """
        Update LLR with a single comparison and check thresholds.

        Returns the decision after this update.
        """
        self._samples_evaluated += 1

        if child_score > parent_score:
            self._wins += 1
            self._llr += self.llr_win
        elif child_score < parent_score:
            self._losses += 1
            self._llr += self.llr_loss
        else:
            self._ties += 1
            # Ties contribute nothing to LLR

        # Check thresholds
        if self._llr >= self.threshold_accept:
            return SPRTDecision.ACCEPT
        elif self._llr <= self.threshold_reject:
            return SPRTDecision.REJECT
        else:
            return SPRTDecision.UNDECIDED

    def evaluate_initial_batch(
        self,
        parent_scores: Sequence[float],
        child_scores: Sequence[float],
    ) -> SPRTResult:
        """
        Evaluate the initial batch of M samples using SPRT.

        This processes samples one at a time, checking thresholds after each.
        If a decision is reached before all samples are processed, it returns
        early with the number of samples actually used.

        Args:
            parent_scores: Scores of parent candidate on initial batch.
            child_scores: Scores of child candidate on same batch.

        Returns:
            SPRTResult with decision (ACCEPT/REJECT/UNDECIDED) and statistics.
        """
        self.reset()

        if len(parent_scores) != len(child_scores):
            raise ValueError("parent_scores and child_scores must have same length")

        for parent_score, child_score in zip(parent_scores, child_scores, strict=False):
            decision = self._update_llr(parent_score, child_score)

            if decision == SPRTDecision.ACCEPT:
                return SPRTResult(
                    decision=SPRTDecision.ACCEPT,
                    llr=self._llr,
                    samples_evaluated=self._samples_evaluated,
                    wins=self._wins,
                    losses=self._losses,
                    ties=self._ties,
                    reason="early_accept",
                )
            elif decision == SPRTDecision.REJECT:
                return SPRTResult(
                    decision=SPRTDecision.REJECT,
                    llr=self._llr,
                    samples_evaluated=self._samples_evaluated,
                    wins=self._wins,
                    losses=self._losses,
                    ties=self._ties,
                    reason="early_reject",
                )

        # All samples processed, still undecided
        return SPRTResult(
            decision=SPRTDecision.UNDECIDED,
            llr=self._llr,
            samples_evaluated=self._samples_evaluated,
            wins=self._wins,
            losses=self._losses,
            ties=self._ties,
            reason="undecided",
        )

    def continue_evaluation(
        self,
        remaining_sample_ids: list[DataId],
        trainset_fetch: Callable[[list[DataId]], list[DataInst]],
        parent_eval_fn: Callable[[list[DataInst]], list[float]],
        child_eval_fn: Callable[[list[DataInst]], list[float]],
    ) -> SPRTResult:
        """
        Continue SPRT evaluation with additional samples until decision or exhaustion.

        Evaluates samples one at a time to minimize wasted evaluations.

        Args:
            remaining_sample_ids: IDs of samples not yet evaluated.
            trainset_fetch: Function to fetch DataInst from sample IDs.
            parent_eval_fn: Function to evaluate parent on a batch, returns scores.
            child_eval_fn: Function to evaluate child on a batch, returns scores.

        Returns:
            SPRTResult with final decision and statistics.
        """
        additional_evals = 0

        for sample_id in remaining_sample_ids:
            # Fetch and evaluate one sample at a time
            batch = trainset_fetch([sample_id])
            parent_score = parent_eval_fn(batch)[0]
            child_score = child_eval_fn(batch)[0]
            additional_evals += 2  # One eval each for parent and child

            decision = self._update_llr(parent_score, child_score)

            if decision == SPRTDecision.ACCEPT:
                return SPRTResult(
                    decision=SPRTDecision.ACCEPT,
                    llr=self._llr,
                    samples_evaluated=self._samples_evaluated,
                    wins=self._wins,
                    losses=self._losses,
                    ties=self._ties,
                    reason="early_accept",
                )
            elif decision == SPRTDecision.REJECT:
                return SPRTResult(
                    decision=SPRTDecision.REJECT,
                    llr=self._llr,
                    samples_evaluated=self._samples_evaluated,
                    wins=self._wins,
                    losses=self._losses,
                    ties=self._ties,
                    reason="early_reject",
                )

        # All remaining samples exhausted, make greedy decision
        if self._llr > 0:
            return SPRTResult(
                decision=SPRTDecision.ACCEPT,
                llr=self._llr,
                samples_evaluated=self._samples_evaluated,
                wins=self._wins,
                losses=self._losses,
                ties=self._ties,
                reason="max_samples",
            )
        else:
            # LLR <= 0, reject (includes LLR == 0 case)
            return SPRTResult(
                decision=SPRTDecision.REJECT,
                llr=self._llr,
                samples_evaluated=self._samples_evaluated,
                wins=self._wins,
                losses=self._losses,
                ties=self._ties,
                reason="max_samples",
            )

    def get_additional_evals_count(self, initial_samples: int) -> int:
        """Return number of additional evaluations beyond initial batch."""
        # Each additional sample requires 2 evals (parent + child)
        additional_samples = self._samples_evaluated - initial_samples
        return max(0, additional_samples * 2)
