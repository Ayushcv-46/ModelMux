"""
Unit tests for Circuit Breaker (Day 28).

Verifies:
1. Initial state is CLOSED with allow() == True.
2. Consecutive failures up to threshold N transition CLOSED -> OPEN.
3. While OPEN, allow() returns False immediately.
4. Success in CLOSED resets the failure count to 0.
5. Error qualification: 5xx, timeouts, and network errors trip the breaker;
   4xx client errors (bad request, auth) do NOT trip the breaker.
6. Rate limit (429) separate treatment behavior.
7. Manual trip, reset, and clock injection.
"""

import httpx
import pytest

from app.errors import (
    ProviderAuthenticationError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)
from app.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    counts_toward_breaker,
)


def test_initial_state_is_closed():
    """Breaker should start in CLOSED state with 0 failures and allow() == True."""
    cb = CircuitBreaker(name="groq", failure_threshold=3)

    assert cb.state == CircuitState.CLOSED
    assert cb.is_closed is True
    assert cb.is_open is False
    assert cb.is_half_open is False
    assert cb.failure_count == 0
    assert cb.allow() is True
    assert cb.opened_at is None


def test_success_resets_failure_count():
    """A success in CLOSED state should reset the consecutive failure counter."""
    cb = CircuitBreaker(name="gemini", failure_threshold=3)

    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.is_closed is True

    # Call succeeds before reaching threshold
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.is_closed is True
    assert cb.allow() is True


def test_closed_to_open_after_n_consecutive_failures():
    """Breaker should trip from CLOSED to OPEN upon reaching N consecutive failures."""
    cb = CircuitBreaker(name="ollama", failure_threshold=3)

    # 1st failure
    tripped = cb.record_failure(ProviderTimeout("Timeout 1"))
    assert tripped is False
    assert cb.failure_count == 1
    assert cb.is_closed is True
    assert cb.allow() is True

    # 2nd failure
    tripped = cb.record_failure(ProviderError("500 Server error", status_code=500))
    assert tripped is False
    assert cb.failure_count == 2
    assert cb.is_closed is True
    assert cb.allow() is True

    # 3rd failure -> Reach threshold of 3
    tripped = cb.record_failure(httpx.ConnectError("Connection refused"))
    assert tripped is True
    assert cb.failure_count == 3
    assert cb.state == CircuitState.OPEN
    assert cb.is_open is True
    assert cb.is_closed is False


def test_while_open_allow_returns_false():
    """While in OPEN state, allow() must return False immediately."""
    cb = CircuitBreaker(name="mock", failure_threshold=2)

    cb.record_failure()
    cb.record_failure()

    assert cb.is_open is True
    assert cb.allow() is False
    assert cb.allow() is False  # Repeated checks still return False


def test_manual_trip_and_reset():
    """trip() forces breaker to OPEN; reset() restores it to clean CLOSED state."""
    cb = CircuitBreaker(name="manual_test", failure_threshold=5)

    cb.trip()
    assert cb.is_open is True
    assert cb.allow() is False
    assert cb.opened_at is not None

    cb.reset()
    assert cb.is_closed is True
    assert cb.failure_count == 0
    assert cb.allow() is True
    assert cb.opened_at is None


def test_qualifying_failures_classification():
    """Verify which exceptions count toward tripping the breaker."""
    # Qualifying (outage indicators)
    assert counts_toward_breaker(ProviderTimeout("Timed out")) is True
    assert counts_toward_breaker(TimeoutError("Asyncio timeout")) is True
    assert counts_toward_breaker(ProviderError("503 crash", status_code=503)) is True
    assert counts_toward_breaker(httpx.ConnectError("Refused")) is True

    req = httpx.Request("POST", "https://api.example.com")
    resp_500 = httpx.Response(status_code=500, request=req)
    assert counts_toward_breaker(httpx.HTTPStatusError("500", request=req, response=resp_500)) is True

    # Non-qualifying (client errors)
    assert counts_toward_breaker(ProviderBadRequest("Bad prompt")) is False
    assert counts_toward_breaker(ProviderAuthenticationError("Bad API key")) is False

    resp_400 = httpx.Response(status_code=400, request=req)
    assert counts_toward_breaker(httpx.HTTPStatusError("400", request=req, response=resp_400)) is False

    resp_401 = httpx.Response(status_code=401, request=req)
    assert counts_toward_breaker(httpx.HTTPStatusError("401", request=req, response=resp_401)) is False


def test_non_qualifying_client_errors_do_not_trip_breaker():
    """Client errors (e.g. 400 Bad Request, 401 Unauthorized) must NOT increment failures or trip breaker."""
    cb = CircuitBreaker(name="gemini", failure_threshold=2)

    cb.record_failure(ProviderBadRequest("Invalid max_tokens"))
    cb.record_failure(ProviderAuthenticationError("Invalid API key"))
    cb.record_failure(ProviderBadRequest("Empty prompt"))

    assert cb.failure_count == 0
    assert cb.is_closed is True
    assert cb.allow() is True


def test_rate_limits_separate_treatment():
    """Rate limit 429 errors should not trip the breaker by default, unless configured."""
    # Default: 429 does NOT trip breaker
    cb_default = CircuitBreaker(name="groq", failure_threshold=2, trip_on_rate_limit=False)
    cb_default.record_failure(ProviderRateLimited("Rate limit hit", retry_after=10.0))
    cb_default.record_failure(ProviderRateLimited("Rate limit hit again", retry_after=5.0))

    assert cb_default.failure_count == 0
    assert cb_default.is_closed is True
    assert cb_default.allow() is True

    # Configured: 429 DOES trip breaker
    cb_strict = CircuitBreaker(name="groq_strict", failure_threshold=2, trip_on_rate_limit=True)
    cb_strict.record_failure(ProviderRateLimited("Rate limit hit"))
    cb_strict.record_failure(ProviderRateLimited("Rate limit hit again"))

    assert cb_strict.failure_count == 2
    assert cb_strict.is_open is True
    assert cb_strict.allow() is False


def test_clock_injection():
    """Clock injection allows deterministic testing of timestamps."""
    fake_time = 1000.0

    def mock_clock():
        return fake_time

    cb = CircuitBreaker(name="clock_test", failure_threshold=1, clock_fn=mock_clock)
    cb.record_failure()

    assert cb.is_open is True
    assert cb.opened_at == 1000.0
    assert cb.last_failure_time == 1000.0


def test_invalid_failure_threshold():
    """failure_threshold must be >= 1."""
    with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
        CircuitBreaker(name="bad", failure_threshold=0)
