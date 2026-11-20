"""
Day 21 — Provider Timeout Unit Tests (asyncio.timeout)

Validates:
1. Slow provider (3.0s) gets cut off at 1.0s timeout and raises ProviderTimeout (504).
2. Elapsed wall-clock time confirms early cancellation (~1.0s, not 3.0s).
3. Fast call succeeds normally without being interrupted.
4. Per-provider timeout values are read from config.yaml.
5. Task cancellation cleanly aborts the sleeping task.
"""

import asyncio
import time
import pytest

from app.config import AppConfig, ProviderConfig, load_yaml_config
from app.errors import ProviderTimeout
from app.providers.mock import MockProvider
from app.reliability.timeout import execute_with_timeout
from app.schemas import ChatRequest, Message


def make_request(text: str = "Hello") -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content=text)])


def test_slow_mock_times_out_at_one_second():
    """
    Core Day 21 requirement:
    Test with a Mock set to 3 seconds and a 1-second limit.
    Output: the slow call fails at about 1 second, not 3.
    """
    async def run():
        provider = MockProvider(min_delay=3.0, max_delay=3.0)
        request = make_request()

        start = time.perf_counter()
        with pytest.raises(ProviderTimeout) as exc_info:
            await execute_with_timeout(provider, request, timeout_seconds=1.0)
        elapsed = time.perf_counter() - start

        # Must fail at ~1.0 second (well before 3.0s)
        assert 0.90 <= elapsed < 2.5
        assert exc_info.value.provider == "mock"
        assert exc_info.value.status_code == 504
        assert "timed out after" in exc_info.value.message

    asyncio.run(run())


def test_fast_call_succeeds_under_timeout():
    """Calls finishing within the timeout deadline return ChatResponse cleanly."""
    async def run():
        provider = MockProvider(min_delay=0.01, max_delay=0.02)
        request = make_request()

        start = time.perf_counter()
        response = await execute_with_timeout(provider, request, timeout_seconds=1.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert response.provider == "mock"
        assert response.text == "This is a mock response."

    asyncio.run(run())


def test_per_provider_timeout_from_config():
    """Verify per-provider timeout values are loaded from config.yaml."""
    cfg = load_yaml_config()

    # Configured values in config.yaml
    assert cfg.get_provider_timeout("groq") == 30.0
    assert cfg.get_provider_timeout("gemini") == 30.0
    assert cfg.get_provider_timeout("ollama") == 60.0
    assert cfg.get_provider_timeout("mock") == 5.0

    # Default fallback for unlisted provider
    assert cfg.get_provider_timeout("unknown_provider", default=12.5) == 12.5


def test_cancellation_behavior():
    """Verify that when a timeout triggers, the provider task is cancelled."""
    was_cancelled = False

    class SlowCancellableProvider(MockProvider):
        async def complete(self, request):
            nonlocal was_cancelled
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                was_cancelled = True
                raise

    async def run():
        provider = SlowCancellableProvider()
        request = make_request()

        with pytest.raises(ProviderTimeout):
            await execute_with_timeout(provider, request, timeout_seconds=0.1)

        assert was_cancelled is True

    asyncio.run(run())
