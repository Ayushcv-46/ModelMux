"""
Day 25 — Retry Budget Unit Tests

Validates:
1. A slow, failing provider stops retrying when total_timeout_seconds is spent.
2. Individual attempts are bounded by the remaining budget, not a fresh full timeout.
3. Backoff sleep stops immediately if the delay would exceed the remaining budget.
4. Fast calls succeed cleanly well within the budget.
5. Configuration loads reliability.total_timeout_seconds properly from config.yaml.
"""

import asyncio
import time
import pytest

from app.config import load_yaml_config
from app.errors import ProviderError, ProviderTimeout
from app.providers.mock import MockProvider
from app.reliability.retry import retry
from app.schemas import ChatRequest, ChatResponse, Message


def make_request(text: str = "Hello") -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content=text)])


# ---------------------------------------------------------------------------
# 1. Total Timeout / Budget Tests
# ---------------------------------------------------------------------------

def test_slow_failing_mock_cuts_off_at_total_budget():
    """
    Day 25 Core Goal:
    Test with a slow, failing Mock and verify the request ends by the budget with a clean error.
    Each attempt takes 0.6s and fails. With 5 attempts and total_timeout=1.2s,
    it must stop after ~2 attempts (~1.2s), NOT running all 5 attempts (3.0s).
    """
    provider = MockProvider(
        min_delay=0.6,
        max_delay=0.6,
        failure_rate=1.0,
        error_kind="500",
    )
    request = make_request()
    call_count = 0

    async def call():
        nonlocal call_count
        call_count += 1
        return await provider.complete(request)

    async def run():
        start = time.perf_counter()
        with pytest.raises(ProviderTimeout) as exc_info:
            await retry(
                call,
                attempts=5,
                base_delay=0.0,  # no extra backoff wait to test execution budget
                total_timeout_seconds=1.2,
            )
        elapsed = time.perf_counter() - start

        # Must have stopped near the 1.2s budget mark (well under 5 * 0.6 = 3.0s)
        assert 1.0 <= elapsed < 2.0
        assert call_count <= 3
        assert "budget" in str(exc_info.value.message).lower() or "timed out" in str(exc_info.value.message).lower()

    asyncio.run(run())


def test_backoff_sleep_aborts_if_delay_exceeds_budget():
    """
    If the calculated backoff delay would exceed the remaining budget,
    the retry loop stops immediately rather than waiting uselessly.
    """
    recorded_sleeps: list[float] = []

    async def fake_sleep(duration: float):
        recorded_sleeps.append(duration)

    call_count = 0

    async def flaky_call():
        nonlocal call_count
        call_count += 1
        raise ProviderError("Temporary 503", status_code=503)

    async def run():
        start = time.perf_counter()
        with pytest.raises(ProviderError):
            await retry(
                flaky_call,
                attempts=5,
                base_delay=10.0,          # 10s backoff delay!
                total_timeout_seconds=2.0, # only 2.0s total budget
                sleep_fn=fake_sleep,
            )
        elapsed = time.perf_counter() - start

        # Proves it stopped immediately after attempt 1 because 10.0s > remaining budget!
        assert call_count == 1
        assert recorded_sleeps == []
        assert elapsed < 1.0

    asyncio.run(run())


def test_fast_call_succeeds_well_under_budget():
    """Fast call completing well within the budget returns normally."""
    async def fast_call():
        return ChatResponse(
            text="Instant answer",
            provider="mock",
            model="mock-1",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1.0,
        )

    async def run():
        res = await retry(
            fast_call,
            attempts=3,
            total_timeout_seconds=10.0,
        )
        assert res.text == "Instant answer"

    asyncio.run(run())


def test_config_loads_reliability_budget():
    """Verify total_timeout_seconds is loaded from config.yaml."""
    cfg = load_yaml_config()
    assert cfg.reliability.total_timeout_seconds == 30.0
    assert cfg.reliability.max_attempts == 3
    assert cfg.reliability.base_delay == 0.1
    assert cfg.reliability.max_delay == 10.0
