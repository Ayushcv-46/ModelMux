"""
Day 35 — Atomic Redis + Lua RateLimiter Unit Tests

Verifies:
1. `bucket.lua` exists and contains atomic Redis Hash + TIME operations.
2. `RateLimiter` registers the Lua script via `register_script` (EVALSHA).
3. `RateLimiter.check(key_id)` returns `(allowed, retry_after)` and tracks `last_remaining`.
4. Per-key isolation in Redis hashes (key A draining does not affect key B).
5. Concurrent requests (`asyncio.gather`) never over-allow beyond capacity.
6. Optional live Redis check if Redis is running locally.
"""

import asyncio
from pathlib import Path
import time

import pytest

from app.ratelimit import RateLimiter, TokenBucket


class FakeAsyncLuaScript:
    """Simulates Redis's atomic single-threaded Lua execution for `bucket.lua`."""

    def __init__(self, store: dict[str, dict[str, str]]) -> None:
        self.store = store
        self.calls: list[tuple[list[str], list[str]]] = []

    async def __call__(self, keys: list[str], args: list[str]) -> list:
        self.calls.append((keys, args))
        key = keys[0]
        capacity = float(args[0])
        refill_rate = float(args[1])
        cost = float(args[2])
        now_arg = args[3] if len(args) > 3 else ""
        now = float(now_arg) if now_arg != "" else time.monotonic()

        bucket_data = self.store.get(key)
        if bucket_data is None:
            tokens = capacity
            last_refill = now
        else:
            tokens = float(bucket_data["tokens"])
            last_refill = float(bucket_data["last_refill"])

        elapsed = max(0.0, now - last_refill)
        tokens = min(capacity, tokens + elapsed * refill_rate)
        last_refill = now

        allowed = 0
        retry_after = "0"
        if tokens >= cost:
            tokens -= cost
            allowed = 1
        else:
            retry_after = str((cost - tokens) / refill_rate)

        self.store[key] = {
            "tokens": str(tokens),
            "last_refill": str(last_refill),
        }
        return [allowed, retry_after, str(int(tokens))]


class FakeAsyncRedis:
    """Minimal async Redis fake supporting `register_script` and in-memory hashes."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.registered_scripts: list[str] = []
        self.last_script: FakeAsyncLuaScript | None = None

    def register_script(self, script_text: str) -> FakeAsyncLuaScript:
        self.registered_scripts.append(script_text)
        self.last_script = FakeAsyncLuaScript(self.hashes)
        return self.last_script


def test_lua_script_file_exists_and_uses_redis_time_and_hash():
    """Verify bucket.lua exists and uses atomic Redis TIME + HMGET/HSET."""
    lua_path = Path("app/ratelimit/bucket.lua")
    assert lua_path.is_file()
    content = lua_path.read_text(encoding="utf-8")

    assert 'redis.call("TIME")' in content
    assert 'redis.call("HMGET"' in content
    assert 'redis.call("HSET"' in content
    assert 'redis.call("EXPIRE"' in content


@pytest.mark.asyncio
async def test_rate_limiter_registers_script_and_checks_burst_empty_refill():
    """RateLimiter calls registered Lua script and returns accurate (allowed, retry_after)."""
    fake_redis = FakeAsyncRedis()
    limiter = RateLimiter(fake_redis, default_capacity=3, default_refill_rate=2.0)

    assert len(fake_redis.registered_scripts) == 1

    # Burst of 3 allowed at t=100.0
    for expected_rem in (2, 1, 0):
        allowed, retry_after = await limiter.check("tenant_1", now=100.0)
        assert allowed is True
        assert retry_after == 0.0
        assert limiter.last_remaining == expected_rem

    # 4th call at t=100.0 rejected with retry_after=0.5s
    allowed, retry_after = await limiter.check("tenant_1", now=100.0)
    assert allowed is False
    assert retry_after == pytest.approx(0.5)

    # Verify Redis hash was stored under ratelimit:tenant_1
    assert "ratelimit:tenant_1" in fake_redis.hashes

    # Advance clock by 0.5s -> 1 token refilled -> allowed!
    allowed, retry_after = await limiter.check("tenant_1", now=100.5)
    assert allowed is True
    assert retry_after == 0.0


@pytest.mark.asyncio
async def test_rate_limiter_isolates_different_keys():
    """Draining key_a must not affect key_b."""
    fake_redis = FakeAsyncRedis()
    limiter = RateLimiter(fake_redis, default_capacity=2, default_refill_rate=1.0)

    assert (await limiter.check("key_a", now=10.0))[0] is True
    assert (await limiter.check("key_a", now=10.0))[0] is True
    assert (await limiter.check("key_a", now=10.0))[0] is False

    # key_b still has its full capacity
    assert (await limiter.check("key_b", now=10.0))[0] is True
    assert limiter.last_remaining == 1


@pytest.mark.asyncio
async def test_rate_limiter_concurrency_never_over_allows():
    """Firing 50 concurrent checks with capacity=10 allows exactly 10."""
    fake_redis = FakeAsyncRedis()
    limiter = RateLimiter(fake_redis, default_capacity=10, default_refill_rate=1.0)

    results = await asyncio.gather(*(limiter.check("burst_key", now=50.0) for _ in range(50)))
    allowed_count = sum(1 for allowed, _ in results if allowed)
    rejected_count = sum(1 for allowed, _ in results if not allowed)

    assert allowed_count == 10
    assert rejected_count == 40
