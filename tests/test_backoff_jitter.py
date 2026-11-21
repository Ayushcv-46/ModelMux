"""
Day 24 — Backoff & Jitter Unit Tests

Validates:
1. Exponential backoff delays grow with attempt count: base * (2 ** (attempt - 1)).
2. Delays are capped at max_delay.
3. Randomized capped jitter prevents synchronized retries across concurrent clients.
4. Sleep function is injectable so retry tests run instantaneously without real sleeps.
5. Upstream Retry-After is respected when greater than the calculated delay.
"""

import asyncio
import random
import pytest

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.reliability.retry import calculate_backoff, retry
from app.schemas import ChatResponse


# ---------------------------------------------------------------------------
# 1. Backoff Formula & Capping Tests
# ---------------------------------------------------------------------------

def test_exponential_backoff_delays_grow():
    """Without jitter, delays double with each successive attempt."""
    base = 1.0
    delays = [
        calculate_backoff(attempt=i, base_delay=base, max_delay=60.0, jitter=False)
        for i in range(1, 6)
    ]
    # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]
    # Assert each step is strictly greater than the previous
    for i in range(len(delays) - 1):
        assert delays[i] < delays[i + 1]


def test_exponential_backoff_is_capped_at_max_delay():
    """Delays never exceed max_delay regardless of attempt count."""
    max_d = 10.0
    delays = [
        calculate_backoff(attempt=i, base_delay=1.0, max_delay=max_d, jitter=False)
        for i in range(1, 12)
    ]
    # All later attempts must be exactly capped at 10.0
    assert all(d <= max_d for d in delays)
    assert delays[-1] == max_d


def test_zero_base_delay_returns_zero():
    """If base_delay is 0, delay is always 0.0."""
    assert calculate_backoff(attempt=3, base_delay=0.0) == 0.0


# ---------------------------------------------------------------------------
# 2. Randomized Jitter Tests (Engineering Refinement)
# ---------------------------------------------------------------------------

def test_randomized_capped_jitter_bounds():
    """Jittered delays fall within [0.5 * raw_delay, raw_delay]."""
    base = 2.0
    # Attempt 3: raw_delay = 2.0 * (2 ** 2) = 8.0
    # Capped jitter range: [4.0, 8.0]
    for _ in range(50):
        val = calculate_backoff(attempt=3, base_delay=base, jitter=True)
        assert 4.0 <= val <= 8.0


def test_assert_no_synchronized_retries():
    """
    Deterministic Engineering Refinement:
    Two concurrent clients with independent RNGs produce different retry timings,
    proving retries will not collide / synchronize into a 'Thundering Herd'.
    """
    rng_client_a = random.Random(42)
    rng_client_b = random.Random(1337)

    delays_a = [
        calculate_backoff(attempt=i, base_delay=1.0, rng=rng_client_a)
        for i in range(1, 6)
    ]
    delays_b = [
        calculate_backoff(attempt=i, base_delay=1.0, rng=rng_client_b)
        for i in range(1, 6)
    ]

    # The sequences must not be identical
    assert delays_a != delays_b
    # Every pair at the same attempt must be decorrelated
    for d_a, d_b in zip(delays_a, delays_b):
        assert d_a != d_b


# ---------------------------------------------------------------------------
# 3. Injectable Sleep Tests (Zero-Latency Verification)
# ---------------------------------------------------------------------------

def test_retry_uses_injectable_sleep_with_growing_delays():
    """Verify that retry() calls the injected sleep_fn with growing delays."""
    recorded_sleeps: list[float] = []

    async def fake_sleep(duration: float):
        recorded_sleeps.append(duration)

    call_count = 0

    async def flaky_call():
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            raise ProviderError("Temporary 500", status_code=500)
        return ChatResponse(
            text="Success",
            provider="mock",
            model="mock-1",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1.0,
        )

    async def run():
        res = await retry(
            flaky_call,
            attempts=4,
            base_delay=1.0,
            jitter=False,  # deterministic growth: 1.0, 2.0, 4.0
            sleep_fn=fake_sleep,
        )
        assert res.text == "Success"
        assert call_count == 4
        # We failed 3 times, so we slept 3 times with 1.0, 2.0, 4.0
        assert recorded_sleeps == [1.0, 2.0, 4.0]

    asyncio.run(run())


def test_retry_respects_retry_after_override():
    """If provider signals a Retry-After greater than backoff, wait at least that long."""
    recorded_sleeps: list[float] = []

    async def fake_sleep(duration: float):
        recorded_sleeps.append(duration)

    call_count = 0

    async def rate_limited_call():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 429 with Retry-After: 7.5 seconds (greater than base_delay 1.0s)
            raise ProviderRateLimited("Rate limited", provider="mock", retry_after=7.5)
        return ChatResponse(
            text="Success",
            provider="mock",
            model="mock-1",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1.0,
        )

    async def run():
        res = await retry(
            rate_limited_call,
            attempts=3,
            base_delay=1.0,
            jitter=False,
            sleep_fn=fake_sleep,
        )
        assert res.text == "Success"
        # Sleep was bumped to 7.5s due to Retry-After!
        assert recorded_sleeps == [7.5]

    asyncio.run(run())
