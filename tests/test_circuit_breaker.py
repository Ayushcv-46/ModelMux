"""
Unit tests for Circuit Breaker (Days 28 & 29).

Verifies:
1. Initial state is CLOSED with allow() == True.
2. Consecutive failures up to threshold N transition CLOSED -> OPEN.
3. While OPEN, allow() returns False immediately.
4. Success in CLOSED resets the failure count to 0.
5. Error qualification: 5xx, timeouts, and network errors trip the breaker;
   4xx client errors (bad request, auth) do NOT trip the breaker.
6. Rate limit (429) separate treatment behavior.
7. Manual trip, reset, and clock injection.
8. Day 29: Cool-down timer transitions OPEN -> HALF-OPEN.
9. Day 29: In HALF-OPEN, exactly ONE trial probe is granted (concurrency safe).
10. Day 29: Trial probe success closes breaker; trial probe failure re-opens it.
11. Day 29: Fallback loop integration: failing provider gets skipped, then recovers when fixed.
"""

import asyncio
import httpx
import pytest

from app.config import AppConfig, ProviderConfig, ReliabilityConfig
from app.errors import (
    AllProvidersFailed,
    CircuitBreakerOpen,
    ProviderAuthenticationError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)
from app.providers.mock import MockProvider
from app.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    counts_toward_breaker,
)
from app.reliability.fallback import execute_fallback_chain
from app.schemas import ChatRequest, Message


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


def test_invalid_parameters():
    """Verify bounds validation for threshold and recovery_timeout."""
    with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
        CircuitBreaker(name="bad", failure_threshold=0)

    with pytest.raises(ValueError, match="recovery_timeout must be non-negative"):
        CircuitBreaker(name="bad", recovery_timeout=-1.0)


# ============================================================================
# Day 29 Tests: Cool-down, HALF-OPEN trial probes, recovery & fallback loop
# ============================================================================

def test_cooldown_timer_transitions_open_to_half_open():
    """After recovery_timeout elapses, breaker automatically moves from OPEN to HALF_OPEN."""
    current_time = 100.0

    def fake_clock():
        return current_time

    cb = CircuitBreaker(
        name="cooldown_test",
        failure_threshold=1,
        recovery_timeout=30.0,
        clock_fn=fake_clock,
    )

    # Trip the breaker at t=100
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open is True
    assert cb.allow() is False

    # Advance time by 15s (still within cool-down)
    current_time = 115.0
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False

    # Advance time to t=130 (cool-down period expired)
    current_time = 130.0
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.is_half_open is True


def test_half_open_allows_exactly_one_trial_probe():
    """In HALF_OPEN, allow() returns True for the 1st probe, then False for subsequent calls."""
    current_time = 100.0

    def fake_clock():
        return current_time

    cb = CircuitBreaker(
        name="probe_test",
        failure_threshold=1,
        recovery_timeout=30.0,
        clock_fn=fake_clock,
    )

    # Trip breaker
    cb.record_failure()
    assert cb.is_open is True

    # Advance past cool-down -> enters HALF_OPEN
    current_time = 130.0
    assert cb.state == CircuitState.HALF_OPEN

    # 1st call gets the trial probe
    assert cb.allow() is True

    # 2nd and 3rd concurrent calls while probe is in flight must get False
    assert cb.allow() is False
    assert cb.allow() is False


def test_trial_probe_success_closes_breaker():
    """If the trial probe in HALF_OPEN succeeds, breaker transitions to CLOSED."""
    current_time = 100.0

    def fake_clock():
        return current_time

    cb = CircuitBreaker(
        name="recovery_test",
        failure_threshold=1,
        recovery_timeout=20.0,
        clock_fn=fake_clock,
    )

    # Trip breaker and advance past cool-down
    cb.record_failure()
    current_time = 125.0
    assert cb.state == CircuitState.HALF_OPEN

    # Probe granted and succeeds
    assert cb.allow() is True
    cb.record_success()

    # Breaker fully recovers to CLOSED
    assert cb.state == CircuitState.CLOSED
    assert cb.is_closed is True
    assert cb.failure_count == 0
    assert cb.opened_at is None
    # Subsequent calls allowed normally
    assert cb.allow() is True


def test_trial_probe_failure_reopens_breaker_and_resets_cooldown():
    """If the trial probe in HALF_OPEN fails, breaker re-trips to OPEN with fresh cool-down."""
    current_time = 100.0

    def fake_clock():
        return current_time

    cb = CircuitBreaker(
        name="reopen_test",
        failure_threshold=1,
        recovery_timeout=30.0,
        clock_fn=fake_clock,
    )

    # Trip breaker at t=100
    cb.record_failure()

    # Advance to t=130 -> moves to HALF_OPEN
    current_time = 130.0
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow() is True  # probe granted

    # Trial probe fails at t=132 with 500 error
    current_time = 132.0
    tripped = cb.record_failure(ProviderError("500 Still crashed", status_code=500))
    assert tripped is True

    # Re-tripped to OPEN with fresh opened_at timestamp (132.0)
    assert cb.state == CircuitState.OPEN
    assert cb.opened_at == 132.0
    assert cb.allow() is False

    # At t=150 (only 18s after re-trip): still cooling down
    current_time = 150.0
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False

    # At t=162 (30s after re-trip): moves to HALF_OPEN again
    current_time = 162.0
    assert cb.state == CircuitState.HALF_OPEN


def test_concurrent_probes_guarantee_at_most_one():
    """Under concurrency, exactly one coroutine acquires the trial probe in HALF_OPEN."""
    cb = CircuitBreaker(name="concurrent_test", failure_threshold=1, recovery_timeout=0.0)
    cb.trip()
    # recovery_timeout=0.0 means it is immediately eligible for HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    async def run_concurrent():
        results = await asyncio.gather(*(asyncio.to_thread(cb.allow) for _ in range(50)))
        # Exactly one call should be granted True, rest 49 False
        assert results.count(True) == 1
        assert results.count(False) == 49

    asyncio.run(run_concurrent())


def test_fallback_loop_skips_open_provider_then_recovers_when_fixed():
    """
    Day 29 Core End-to-End Goal:
    Provider A fails repeatedly -> breaker trips to OPEN.
    While OPEN, Provider A is skipped in 0ms, falling back to Provider B.
    After cool-down, Provider A is given a trial probe in HALF_OPEN, succeeds,
    and returns to normal service!
    """
    current_time = 1000.0

    def fake_clock():
        return current_time

    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "failing_prov_a"

    provider_b = MockProvider(failure_rate=0.0, reply="Provider B backup reply", min_delay=0.0, max_delay=0.0)
    provider_b.name = "healthy_prov_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=1,
            base_delay=0.0,
            total_timeout_seconds=5.0,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=30.0,
        ),
        providers={
            "failing_prov_a": ProviderConfig(timeout_seconds=1.0),
            "healthy_prov_b": ProviderConfig(timeout_seconds=1.0),
        },
    )

    cb_a = CircuitBreaker(
        name="failing_prov_a",
        failure_threshold=2,
        recovery_timeout=30.0,
        clock_fn=fake_clock,
    )
    cb_b = CircuitBreaker(
        name="healthy_prov_b",
        failure_threshold=2,
        recovery_timeout=30.0,
        clock_fn=fake_clock,
    )
    breakers = {"failing_prov_a": cb_a, "healthy_prov_b": cb_b}

    def make_req():
        return ChatRequest(messages=[Message(role="user", content="Test")])

    async def run():
        nonlocal current_time

        # Call 1: Provider A fails (1st failure) -> fallback to B
        resp1 = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_req(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
        )
        assert resp1.provider == "healthy_prov_b"
        assert cb_a.failure_count == 1
        assert cb_a.is_closed is True

        # Call 2: Provider A fails (2nd failure) -> Reaches threshold 2 -> Breaker A trips to OPEN!
        resp2 = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_req(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
        )
        assert resp2.provider == "healthy_prov_b"
        assert cb_a.failure_count == 2
        assert cb_a.is_open is True

        # Call 3: Provider A's breaker is OPEN! Provider A is SKIPPED in 0ms!
        attempts: list[dict] = []
        resp3 = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_req(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
            attempts_log=attempts,
        )
        assert resp3.provider == "healthy_prov_b"
        # Since Provider A was skipped by circuit breaker without running network attempts,
        # attempts log only contains provider B's successful attempt!
        assert len(attempts) == 1
        assert attempts[0]["provider"] == "healthy_prov_b"
        assert attempts[0]["status"] == 200

        # Advance time by 35s (past 30s cool-down)
        current_time += 35.0
        assert cb_a.state == CircuitState.HALF_OPEN

        # Fix Provider A!
        provider_a.failure_rate = 0.0
        provider_a.reply = "Provider A is back online!"

        # Call 4: Provider A is in HALF_OPEN -> trial probe allowed -> succeeds!
        resp4 = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_req(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
        )
        assert resp4.provider == "failing_prov_a"
        assert resp4.text == "Provider A is back online!"
        assert cb_a.is_closed is True
        assert cb_a.failure_count == 0

        # Call 5: Provider A is healthy and CLOSED -> answers directly
        resp5 = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_req(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
        )
        assert resp5.provider == "failing_prov_a"

    asyncio.run(run())
