# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ResidualWeightedSampler:
    """Weighted sampler using residual accumulation (error diffusion).

    Unlike probability-based sampling where low-weight samples may be perpetually
    skipped, this algorithm guarantees all samples are eventually visited.
    Weight determines expected frequency, not just selection probability.

    Algorithm:
        For each element in round-robin order:
        1. accumulator += weight
        2. output = floor(accumulator)
        3. accumulator -= output (keep fractional remainder)
        4. emit element `output` times

    Properties:
        - weight >= 1: sampled at least once per round
        - weight = 0.5: sampled once every 2 rounds (deterministically)
        - weight = 2.0: sampled twice per round (on average)
        - weight = 0: never sampled
        - Dynamic weight updates preserve accumulator state for smooth transitions

    Example:
        >>> sampler = ResidualWeightedSampler(3)
        >>> sampler.update_weights([1.0, 0.5, 2.0])
        >>> sampler.sample(7)  # sum(weights) = 3.5, so ~2 rounds
        [0, 2, 2, 1, 0, 2, 2]
    """

    n: int

    _weights: list[float] = field(default_factory=list)
    _accumulators: list[float] = field(default_factory=list)
    _cursor: int = field(default=0)
    _buffer: deque[int] = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Initialize weights and accumulators."""
        self._weights = [1.0] * self.n
        self._accumulators = [0.0] * self.n

    def update_weights(self, weights: list[float]) -> None:
        """Update weights while preserving accumulator state.

        This allows smooth transitions when weights change dynamically.
        The accumulator state is preserved, so partial progress toward
        the next sample is not lost.

        Args:
            weights: New weights for each element. Must have length n.

        Raises:
            ValueError: If weights length doesn't match n.
        """
        if len(weights) != self.n:
            raise ValueError(f"Weights length must be {self.n}, got {len(weights)}")
        self._weights = list(weights)

    def sample(self, k: int = 1) -> list[int]:
        """Sample k element indices according to weights.

        Elements with higher weights appear more frequently. Unlike probability-based
        sampling, this guarantees eventual coverage of all non-zero weight elements.

        Args:
            k: Number of samples to return.

        Returns:
            List of k element indices (0 to n-1). May contain duplicates if
            an element has weight > 1.
        """
        result: list[int] = []

        # First consume any buffered elements from previous sampling
        while k > 0 and self._buffer:
            result.append(self._buffer.popleft())
            k -= 1

        if k == 0:
            return result

        # Generate more elements by traversing the list
        while k > 0:
            idx = self._cursor
            w = self._weights[idx]

            # Core algorithm: accumulate weight and emit floor(accumulator) times
            self._accumulators[idx] += w
            count = int(self._accumulators[idx])
            self._accumulators[idx] -= count

            # Distribute produced elements
            for _ in range(count):
                if k > 0:
                    result.append(idx)
                    k -= 1
                else:
                    # Buffer excess for next sample() call
                    self._buffer.append(idx)

            # Move cursor (round-robin)
            self._cursor = (self._cursor + 1) % self.n

        return result

    def reset(self) -> None:
        """Reset accumulators, cursor, and buffer to initial state.

        Weights are preserved. Use this to start fresh sampling without
        losing the current weight configuration.
        """
        self._accumulators = [0.0] * self.n
        self._cursor = 0
        self._buffer.clear()
