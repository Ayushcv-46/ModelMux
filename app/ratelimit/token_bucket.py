"""
Pure Python Token Bucket rate limiter (Day 34).

Supports:
- Bursting up to `capacity` tokens.
- Steady refilling at `refill_rate` tokens per second via lazy calculation.
- Injectable clock (`now` parameter or `clock_fn`) for deterministic testing.
"""

import time
from typing import Callable


class TokenBucket:
    """In-memory Token Bucket rate limiter."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self.capacity: float = float(capacity)
        self.refill_rate: float = float(refill_rate)
        self.clock_fn: Callable[[], float] = clock_fn
        self.tokens: float = float(capacity)
        self.last_refill: float | None = None

    def allow(self, now: float | None = None, cost: float = 1.0) -> tuple[bool, float]:
        """
        Attempt to consume `cost` tokens at timestamp `now`.

        Returns:
            (allowed, retry_after):
            - allowed=True, retry_after=0.0 if enough tokens were available.
            - allowed=False, retry_after=seconds_to_wait if bucket has insufficient tokens.
        """
        if cost <= 0:
            raise ValueError("cost must be positive")

        current_time = now if now is not None else self.clock_fn()

        if self.last_refill is None:
            self.last_refill = current_time

        elapsed = max(0.0, current_time - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = current_time

        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0

        deficit = cost - self.tokens
        retry_after = deficit / self.refill_rate
        return False, retry_after

    @property
    def remaining(self) -> int:
        """Whole tokens currently available in the bucket."""
        return int(self.tokens)
