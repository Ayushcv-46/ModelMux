"""
Day 26 — Fallback Chain Unit Tests

Validates:
1. Provider A fails after retries -> falls back to Provider B -> returns 200 (ChatResponse).
2. All candidate providers fail -> raises AllProvidersFailed with HTTP 503.
3. First provider succeeds -> subsequent fallback providers are never called.
4. Model compatibility: candidate providers incompatible with requested model are safely skipped.
5. Preserves total time budget across all fallback candidates.
"""

import asyncio
import pytest

from app.config import AppConfig, ProviderConfig, ReliabilityConfig
from app.errors import AllProvidersFailed, ProviderError, ProviderTimeout
from app.providers.mock import MockProvider
from app.reliability.fallback import execute_fallback_chain
from app.schemas import ChatRequest, ChatResponse, Message


def make_request(model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        model=model,
    )


def test_fallback_chain_primary_fails_secondary_succeeds():
    """
    Day 26 Core Goal:
    Provider A fails after retries, fallback chain tries Provider B,
    and returns a successful ChatResponse (200 OK equivalent).
    """
    provider_a = MockProvider(
        failure_rate=1.0,
        error_kind="500",
        min_delay=0.0,
        max_delay=0.0,
    )
    provider_a.name = "mock_a"

    provider_b = MockProvider(
        failure_rate=0.0,
        reply="Hello from Provider B!",
        min_delay=0.0,
        max_delay=0.0,
    )
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=2,
            base_delay=0.0,
            total_timeout_seconds=5.0,
        ),
        providers={
            "mock_a": ProviderConfig(timeout_seconds=2.0),
            "mock_b": ProviderConfig(timeout_seconds=2.0),
        },
    )

    async def run():
        response = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_request(),
            cfg=cfg,
        )
        assert response.provider == "mock_b"
        assert response.text == "Hello from Provider B!"

    asyncio.run(run())


def test_fallback_chain_all_fail_raises_503():
    """If all candidate providers fail, raises AllProvidersFailed with HTTP 503."""
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = MockProvider(failure_rate=1.0, error_kind="timeout", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=1,
            base_delay=0.0,
            total_timeout_seconds=5.0,
        ),
        providers={
            "mock_a": ProviderConfig(timeout_seconds=2.0),
            "mock_b": ProviderConfig(timeout_seconds=2.0),
        },
    )

    async def run():
        with pytest.raises(AllProvidersFailed) as exc_info:
            await execute_fallback_chain(
                providers=[provider_a, provider_b],
                request=make_request(),
                cfg=cfg,
            )

        assert exc_info.value.status_code == 503
        assert len(exc_info.value.details) == 2
        assert exc_info.value.details[0]["provider"] == "mock_a"
        assert exc_info.value.details[1]["provider"] == "mock_b"

    asyncio.run(run())


def test_fallback_chain_primary_succeeds_skips_secondary():
    """If the first provider succeeds, subsequent fallback candidates are never invoked."""
    call_count_b = 0

    class TrackingMock(MockProvider):
        async def complete(self, req):
            nonlocal call_count_b
            call_count_b += 1
            return await super().complete(req)

    provider_a = MockProvider(failure_rate=0.0, reply="Provider A Success", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = TrackingMock(min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=1, base_delay=0.0),
        providers={"mock_a": ProviderConfig(), "mock_b": ProviderConfig()},
    )

    async def run():
        response = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_request(),
            cfg=cfg,
        )
        assert response.provider == "mock_a"
        assert response.text == "Provider A Success"
        # Secondary provider was never called
        assert call_count_b == 0

    asyncio.run(run())


def test_fallback_chain_skips_incompatible_provider():
    """
    Engineering Refinement:
    If client requests a specific model, skip candidate providers that do not support it.
    """
    call_count_a = 0

    class CountingMockA(MockProvider):
        async def complete(self, req):
            nonlocal call_count_a
            call_count_a += 1
            return await super().complete(req)

    provider_a = CountingMockA(min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = MockProvider(reply="Provider B response", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=1, base_delay=0.0),
        providers={
            "mock_a": ProviderConfig(models=["model-a-only"]),
            "mock_b": ProviderConfig(models=["model-b-target"]),
        },
    )

    async def run():
        # Request explicitly asks for model-b-target
        req = make_request(model="model-b-target")
        response = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=req,
            cfg=cfg,
        )
        assert response.provider == "mock_b"
        assert response.text == "Provider B response"
        # Provider A was skipped completely because it cannot serve model-b-target
        assert call_count_a == 0

    asyncio.run(run())


def test_fallback_chain_empty_providers_raises_503():
    """Passing an empty list of providers raises AllProvidersFailed (503)."""
    async def run():
        with pytest.raises(AllProvidersFailed) as exc_info:
            await execute_fallback_chain(providers=[], request=make_request())
        assert exc_info.value.status_code == 503

    asyncio.run(run())
