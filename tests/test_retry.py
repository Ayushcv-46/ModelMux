"""
Day 23 — Retry Unit Tests

Validates:
1. Retrying a function that fails twice with retryable errors succeeds on attempt 3.
2. Stops early on non-retryable errors (e.g. 401 Unauthorized, 400 Bad Request) on attempt 1.
3. Exhausting maximum attempts raises the final error after exactly N calls.
4. Immediate success returns without retrying.
5. Input validation (attempts < 1 raises ValueError).
"""

import asyncio
import pytest

from app.errors import (
    ModelNotSupportedError,
    ProviderAuthenticationError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)
from app.reliability.retry import retry
from app.schemas import ChatResponse


class DeterministicMock:
    """Mock that behaves deterministically on each call based on a predefined sequence."""

    def __init__(self, sequence: list):
        self.sequence = sequence
        self.call_count = 0

    async def call(self):
        self.call_count += 1
        item = self.sequence[self.call_count - 1]
        if isinstance(item, Exception):
            raise item
        return item


def make_response(text: str = "Success") -> ChatResponse:
    return ChatResponse(
        text=text,
        provider="mock",
        model="mock-1",
        tokens_in=5,
        tokens_out=5,
        latency_ms=10.0,
    )


def test_retry_succeeds_on_third_attempt():
    """
    Core Day 23 Goal:
    Mock failing twice then succeeding returns success on try 3.
    """
    mock = DeterministicMock(
        sequence=[
            ProviderTimeout("Temporary timeout", provider="mock"),
            ProviderError("Temporary 503 outage", provider="mock", status_code=503),
            make_response("Success on try 3"),
        ]
    )

    async def run():
        response = await retry(mock.call, attempts=3)
        assert response.text == "Success on try 3"
        assert mock.call_count == 3

    asyncio.run(run())


def test_retry_stops_early_on_non_retryable_error():
    """Non-retryable error (e.g. 401 unauthenticated) aborts immediately on try 1."""
    mock = DeterministicMock(
        sequence=[
            ProviderAuthenticationError("Invalid API key", provider="mock"),
            make_response("Should never reach here"),
        ]
    )

    async def run():
        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await retry(mock.call, attempts=3)

        assert "Invalid API key" in str(exc_info.value)
        # Must have stopped immediately after 1st attempt!
        assert mock.call_count == 1

    asyncio.run(run())


def test_retry_stops_early_on_bad_request():
    """Malformed request (400) or unsupported model aborts immediately on try 1."""
    mock = DeterministicMock(
        sequence=[
            ProviderBadRequest("Malformed JSON", provider="mock"),
            make_response("Should never reach here"),
        ]
    )

    async def run():
        with pytest.raises(ProviderBadRequest):
            await retry(mock.call, attempts=3)

        assert mock.call_count == 1

    asyncio.run(run())


def test_retry_exhaustion_raises_final_error():
    """When all attempts fail with retryable errors, the last error is raised."""
    mock = DeterministicMock(
        sequence=[
            ProviderTimeout("Timeout 1", provider="mock"),
            ProviderRateLimited("Rate limited 2", provider="mock"),
            ProviderError("Server error 3", provider="mock", status_code=502),
        ]
    )

    async def run():
        with pytest.raises(ProviderError) as exc_info:
            await retry(mock.call, attempts=3)

        assert "Server error 3" in str(exc_info.value)
        assert mock.call_count == 3

    asyncio.run(run())


def test_retry_immediate_success_calls_once():
    """A call that succeeds on the first try only runs once."""
    mock = DeterministicMock(sequence=[make_response("Immediate success")])

    async def run():
        response = await retry(mock.call, attempts=3)
        assert response.text == "Immediate success"
        assert mock.call_count == 1

    asyncio.run(run())


def test_retry_invalid_attempts_value():
    """attempts < 1 raises ValueError."""
    async def run():
        with pytest.raises(ValueError):
            await retry(lambda: None, attempts=0)

    asyncio.run(run())
