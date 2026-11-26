"""
Pure Python Token Bucket rate limiter (Day 34).

Supports:
- Bursting up to `capacity` tokens.
- Steady refilling at `refill_rate` tokens per second via lazy calculation.
- Injectable clock (`now` parameter or `clock_fn`) for deterministic testing.
"""

import time
from pathlib import Path
from typing import Any, Callable


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


class RateLimiter:
    """
    Atomic Redis-backed Token Bucket rate limiter using a Lua script (Day 35).

    Uses Redis `register_script` (EVALSHA) to execute read, refill, decrement,
    and write in one atomic server-side step.
    """

    def __init__(
        self,
        redis_client: Any,
        default_capacity: float = 60.0,
        default_refill_rate: float = 1.0,
        key_prefix: str = "ratelimit:",
        lua_path: Path | None = None,
    ) -> None:
        self.redis = redis_client
        self.default_capacity = float(default_capacity)
        self.default_refill_rate = float(default_refill_rate)
        self.key_prefix = key_prefix

        script_path = lua_path or (Path(__file__).resolve().parent / "bucket.lua")
        self.lua_script = script_path.read_text(encoding="utf-8")
        self._script = self.redis.register_script(self.lua_script)
        self.last_remaining: int = int(default_capacity)

    async def check(
        self,
        key_id: str | int,
        capacity: float | None = None,
        refill_rate: float | None = None,
        cost: float = 1.0,
        now: float | None = None,
    ) -> tuple[bool, float]:
        """
        Atomically check and consume `cost` tokens for `key_id` in Redis.

        Returns:
            (allowed, retry_after):
            - allowed: True if request is permitted, False if rate-limited.
            - retry_after: Seconds to wait before retrying (0.0 when allowed).
        """
        cap = float(capacity if capacity is not None else self.default_capacity)
        rate = float(refill_rate if refill_rate is not None else self.default_refill_rate)
        redis_key = f"{self.key_prefix}{key_id}"
        now_arg = "" if now is None else str(now)

        result = await self._script(
            keys=[redis_key],
            args=[str(cap), str(rate), str(cost), now_arg],
        )

        allowed = bool(int(result[0]))
        retry_after = float(result[1])
        self.last_remaining = int(float(result[2])) if len(result) > 2 else 0
        return allowed, retry_after


class InMemoryRateLimiter:
    """
    In-memory fallback rate limiter with the same interface as RateLimiter.
    Used when Redis is unavailable (e.g. local dev/tests without Docker).
    """

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self.last_remaining: int = 0

    async def check(
        self,
        key_id: str | int,
        capacity: float = 60.0,
        refill_rate: float = 1.0,
        cost: float = 1.0,
        now: float | None = None,
    ) -> tuple[bool, float]:
        key = str(key_id)
        bucket = self._buckets.get(key)
        if bucket is None or bucket.capacity != float(capacity) or bucket.refill_rate != float(refill_rate):
            bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
            self._buckets[key] = bucket

        allowed, retry_after = bucket.allow(now=now, cost=cost)
        self.last_remaining = bucket.remaining
        return allowed, retry_after

    def clear(self) -> None:
        self._buckets.clear()
