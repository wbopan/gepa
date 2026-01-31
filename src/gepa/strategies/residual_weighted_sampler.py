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

    def sample(self, k: int = 1, unique: bool = True) -> list[int]:
        """Sample k element indices according to weights.

        Elements with higher weights appear more frequently. Unlike probability-based
        sampling, this guarantees eventual coverage of all non-zero weight elements.

        Args:
            k: Number of samples to return.
            unique: If True (default), each element appears at most once per batch.
                High-weight elements that would be sampled multiple times have their
                excess deferred to subsequent sample() calls via the accumulator.
                If False, elements may appear multiple times in a single batch.

        Returns:
            List of k element indices (0 to n-1). May contain duplicates only if
            unique=False and an element has weight > 1.

        Raises:
            ValueError: If unique=True and k > number of non-zero weight elements.
        """
        if unique:
            non_zero_count = sum(1 for w in self._weights if w > 0)
            if k > non_zero_count:
                raise ValueError(
                    f"Cannot sample {k} unique elements: only {non_zero_count} elements have non-zero weight"
                )

        result: list[int] = []
        seen_in_batch: set[int] = set()

        # First consume any buffered elements from previous sampling (only when unique=False)
        if not unique:
            while k > 0 and self._buffer:
                result.append(self._buffer.popleft())
                k -= 1

        if k == 0:
            return result

        # Generate more elements by traversing the list
        while len(result) < k:
            idx = self._cursor
            w = self._weights[idx]

            # Core algorithm: accumulate weight
            self._accumulators[idx] += w

            # Check if this element should be emitted
            if self._accumulators[idx] >= 1.0:
                if unique:
                    # Unique mode: emit at most once per batch
                    if idx not in seen_in_batch:
                        result.append(idx)
                        seen_in_batch.add(idx)
                        self._accumulators[idx] -= 1.0
                    # If already seen, keep accumulator (defer to next batch)
                else:
                    # Non-unique mode: emit floor(accumulator) times
                    count = int(self._accumulators[idx])
                    self._accumulators[idx] -= count

                    for _ in range(count):
                        if len(result) < k:
                            result.append(idx)
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
