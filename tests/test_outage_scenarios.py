"""
Day 30 — Outage Test Suite (Reliability Milestone v0.3)

Validates ModelMux under four production failure scenarios:
1. Total outage of primary provider -> seamless fallback to secondary provider (200 OK).
2. Flaky / intermittent provider -> self-heals via exponential backoff/retry (200 OK).
3. Slow / hanging provider -> cut off by per-provider timeout, bounded by total budget (200 OK).
4. Catastrophic outage (all providers down) -> returns graceful HTTP 503 AllProvidersFailed with full audit log.

Engineering refinements:
5. Total deadline budget exhaustion across retries and fallbacks.
6. Circuit breaker accelerated failover (0ms skip while OPEN) and cool-down recovery.
7. Full FastAPI /chat endpoint integration test verifying database persistence of attempts log.
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import AppConfig, ProviderConfig, ReliabilityConfig
from app.db import Base, RequestLog, get_db
from app.errors import AllProvidersFailed, CircuitBreakerOpen, ProviderError, ProviderTimeout
from app.main import app, get_provider, get_router
from app.providers.mock import MockProvider
from app.reliability.circuit_breaker import CircuitBreaker, CircuitState
from app.reliability.fallback import execute_fallback_chain
from app.routing.router import Router
from app.routing.strategy import PriorityStrategy
from app.schemas import ChatRequest, Message


def make_request(prompt: str = "Test prompt", model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content=prompt)],
        model=model,
    )


# ============================================================================
# Scenario 1: Total Outage of Primary Provider
# ============================================================================
def test_scenario_1_total_outage_primary_provider_fails_over_to_secondary():
    """
    Scenario 1:
    Primary provider (Provider A) has a 100% 500 internal server error outage.
    Secondary provider (Provider B) is healthy.

    Expected:
    - Gateway attempts Provider A, retries and fails.
    - Fallback kicks in and routes to Provider B.
    - Returns 200 OK with Provider B's response.
    - Attempt log records Provider A failures followed by Provider B success.
    """
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_groq_primary"

    provider_b = MockProvider(failure_rate=0.0, reply="Response from Gemini backup", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_gemini_backup"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=2,
            base_delay=0.0,
            total_timeout_seconds=5.0,
        ),
        providers={
            "mock_groq_primary": ProviderConfig(timeout_seconds=1.0),
            "mock_gemini_backup": ProviderConfig(timeout_seconds=1.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        response = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_request(),
            cfg=cfg,
            attempts_log=attempts,
        )
        assert response.provider == "mock_gemini_backup"
        assert response.text == "Response from Gemini backup"

    asyncio.run(run())

    # Provider A failed 2 attempts, Provider B succeeded on 1 attempt = 3 attempts total
    assert len(attempts) == 3
    assert attempts[0]["provider"] == "mock_groq_primary"
    assert attempts[0]["status"] == 502
    assert attempts[1]["provider"] == "mock_groq_primary"
    assert attempts[1]["status"] == 502
    assert attempts[2]["provider"] == "mock_gemini_backup"
    assert attempts[2]["status"] == 200


# ============================================================================
# Scenario 2: Flaky / Intermittent Provider
# ============================================================================
def test_scenario_2_flaky_provider_self_heals_via_retries():
    """
    Scenario 2:
    Provider A fails its first 2 attempts with transient 502 Bad Gateway errors,
    but recovers on attempt 3.

    Expected:
    - Retries with backoff catch the temporary blip.
    - Request succeeds on try 3 without needing fallback to Provider B.
    - User receives 200 OK from Provider A.
    - Attempts log shows exactly 3 attempts for Provider A.
    """
    attempt_counter = 0

    class FlakyProvider(MockProvider):
        name = "mock_flaky_groq"

        async def complete(self, req: ChatRequest):
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter < 3:
                raise ProviderError(f"Transient 502 error (attempt {attempt_counter})", status_code=502)
            return await super().complete(req)

    flaky_provider = FlakyProvider(failure_rate=0.0, reply="Self-healed on try 3", min_delay=0.0, max_delay=0.0)
    backup_provider = MockProvider(failure_rate=0.0, reply="Backup should not be called", min_delay=0.0, max_delay=0.0)
    backup_provider.name = "mock_backup"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=3,
            base_delay=0.0,
            total_timeout_seconds=5.0,
        ),
        providers={
            "mock_flaky_groq": ProviderConfig(timeout_seconds=1.0),
            "mock_backup": ProviderConfig(timeout_seconds=1.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        response = await execute_fallback_chain(
            providers=[flaky_provider, backup_provider],
            request=make_request(),
            cfg=cfg,
            attempts_log=attempts,
        )
        assert response.provider == "mock_flaky_groq"
        assert response.text == "Self-healed on try 3"

    asyncio.run(run())

    # Verify attempt log: 2 failures followed by success on the same provider
    assert len(attempts) == 3
    assert attempts[0]["provider"] == "mock_flaky_groq"
    assert attempts[0]["status"] == 502
    assert attempts[1]["provider"] == "mock_flaky_groq"
    assert attempts[1]["status"] == 502
    assert attempts[2]["provider"] == "mock_flaky_groq"
    assert attempts[2]["status"] == 200


# ============================================================================
# Scenario 3: Slow / Hanging Provider
# ============================================================================
def test_scenario_3_slow_provider_cut_off_by_timeout_and_falls_back():
    """
    Scenario 3:
    Provider A hangs indefinitely (simulating 10s latency).
    Provider timeout is configured to 0.1s.
    Total timeout budget is 2.0s.

    Expected:
    - Provider A is cut off via ProviderTimeout at 0.1s.
    - Fallback chain switches to responsive Provider B.
    - Returns 200 OK from Provider B in ~0.15s (well within the 2.0s budget).
    - Attempts log documents ProviderTimeout on Provider A.
    """
    provider_a = MockProvider(failure_rate=0.0, min_delay=5.0, max_delay=5.0)  # Slow hanging provider
    provider_a.name = "mock_slow_prov"

    provider_b = MockProvider(failure_rate=0.0, reply="Fast fallback response", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_fast_prov"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=1,
            base_delay=0.0,
            total_timeout_seconds=2.0,
        ),
        providers={
            "mock_slow_prov": ProviderConfig(timeout_seconds=0.1),
            "mock_fast_prov": ProviderConfig(timeout_seconds=1.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        start_t = time.perf_counter()
        response = await execute_fallback_chain(
            providers=[provider_a, provider_b],
            request=make_request(),
            cfg=cfg,
            attempts_log=attempts,
        )
        elapsed = time.perf_counter() - start_t

        assert response.provider == "mock_fast_prov"
        assert response.text == "Fast fallback response"
        # Bounded latency: must finish quickly, not wait 5 seconds!
        assert elapsed < 1.0

    asyncio.run(run())

    assert len(attempts) == 2
    assert attempts[0]["provider"] == "mock_slow_prov"
    assert attempts[0]["status"] == 504
    assert attempts[0]["error_category"] == "ProviderTimeout"
    assert attempts[1]["provider"] == "mock_fast_prov"
    assert attempts[1]["status"] == 200


# ============================================================================
# Scenario 4: Catastrophic Total Outage (All Upstreams Down)
# ============================================================================
def test_scenario_4_total_catastrophic_outage_raises_503():
    """
    Scenario 4:
    Every candidate provider in the system is completely offline.

    Expected:
    - Provider A fails and exhausts retries.
    - Provider B fails and exhausts retries.
    - Gateway raises AllProvidersFailed with status code 503.
    - Attempts log contains full failure trail for every attempted candidate.
    """
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_dead_a"

    provider_b = MockProvider(failure_rate=1.0, error_kind="timeout", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_dead_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=2,
            base_delay=0.0,
            total_timeout_seconds=5.0,
        ),
        providers={
            "mock_dead_a": ProviderConfig(timeout_seconds=1.0),
            "mock_dead_b": ProviderConfig(timeout_seconds=1.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        with pytest.raises(AllProvidersFailed) as exc_info:
            await execute_fallback_chain(
                providers=[provider_a, provider_b],
                request=make_request(),
                cfg=cfg,
                attempts_log=attempts,
            )
        assert exc_info.value.status_code == 503
        assert len(exc_info.value.details) == 2

    asyncio.run(run())

    # 2 attempts for dead_a + 2 attempts for dead_b = 4 attempts total
    assert len(attempts) == 4
    assert attempts[0]["provider"] == "mock_dead_a"
    assert attempts[0]["status"] == 502
    assert attempts[1]["provider"] == "mock_dead_a"
    assert attempts[2]["provider"] == "mock_dead_b"
    assert attempts[2]["status"] == 504
    assert attempts[3]["provider"] == "mock_dead_b"


# ============================================================================
# Scenario 5: Deadline Budget Exhaustion Across Retries & Fallbacks
# ============================================================================
def test_scenario_5_deadline_budget_exhaustion_stops_future_providers():
    """
    Scenario 5:
    Simulates clock advancement: Provider A exhausts the total request deadline.
    Provider B should NEVER be attempted because remaining budget <= 0.
    """
    fake_time = 0.0

    def fake_clock():
        return fake_time

    async def fake_sleep(duration: float):
        nonlocal fake_time
        fake_time += duration

    class ExpiringProvider(MockProvider):
        name = "mock_budget_burner"

        async def complete(self, req: ChatRequest):
            nonlocal fake_time
            fake_time += 1.5  # Burns 1.5s per attempt
            raise ProviderError("Burned time", status_code=500)

    provider_a = ExpiringProvider()
    provider_b = MockProvider(failure_rate=0.0, reply="Never called", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_unreached"

    cfg = AppConfig(
        reliability=ReliabilityConfig(
            max_attempts=3,
            base_delay=0.1,
            total_timeout_seconds=2.0,  # Total budget is 2.0s
        ),
        providers={
            "mock_budget_burner": ProviderConfig(timeout_seconds=5.0),
            "mock_unreached": ProviderConfig(timeout_seconds=5.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        with pytest.raises(AllProvidersFailed):
            await execute_fallback_chain(
                providers=[provider_a, provider_b],
                request=make_request(),
                cfg=cfg,
                sleep_fn=fake_sleep,
                clock_fn=fake_clock,
                attempts_log=attempts,
            )

    asyncio.run(run())

    # Provider B was never attempted because budget expired!
    attempted_providers = {att["provider"] for att in attempts}
    assert "mock_unreached" not in attempted_providers


# ============================================================================
# Scenario 6: Circuit Breaker Fast-Fail & Automatic Recovery
# ============================================================================
def test_scenario_6_circuit_breaker_fast_fail_and_recovery():
    """
    Scenario 6:
    Proves that repeated outages trip the circuit breaker, skipping Provider A in 0ms,
    and recovers smoothly after the cool-down timer.
    """
    current_time = 100.0

    def fake_clock():
        return current_time

    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "groq_outage"

    provider_b = MockProvider(failure_rate=0.0, reply="Gemini backup", min_delay=0.0, max_delay=0.0)
    provider_b.name = "gemini_backup"

    cb_a = CircuitBreaker(name="groq_outage", failure_threshold=2, recovery_timeout=30.0, clock_fn=fake_clock)
    cb_b = CircuitBreaker(name="gemini_backup", failure_threshold=2, recovery_timeout=30.0, clock_fn=fake_clock)
    breakers = {"groq_outage": cb_a, "gemini_backup": cb_b}

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=1, base_delay=0.0, total_timeout_seconds=5.0),
        providers={
            "groq_outage": ProviderConfig(timeout_seconds=1.0),
            "gemini_backup": ProviderConfig(timeout_seconds=1.0),
        },
    )

    async def run():
        nonlocal current_time

        # Fail twice -> trips breaker to OPEN
        await execute_fallback_chain([provider_a, provider_b], make_request(), cfg=cfg, clock_fn=fake_clock, breakers=breakers)
        await execute_fallback_chain([provider_a, provider_b], make_request(), cfg=cfg, clock_fn=fake_clock, breakers=breakers)
        assert cb_a.is_open is True

        # Call 3: Breaker is OPEN -> provider_a skipped in 0ms!
        attempts: list[dict] = []
        resp = await execute_fallback_chain(
            [provider_a, provider_b],
            make_request(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
            attempts_log=attempts,
        )
        assert resp.provider == "gemini_backup"
        assert len(attempts) == 1
        assert attempts[0]["provider"] == "gemini_backup"

        # Advance past 30s cool-down and heal provider_a
        current_time += 35.0
        assert cb_a.state == CircuitState.HALF_OPEN
        provider_a.failure_rate = 0.0
        provider_a.reply = "Groq is restored!"

        # Call 4: Trial probe succeeds -> restores breaker to CLOSED
        recovered_resp = await execute_fallback_chain(
            [provider_a, provider_b],
            make_request(),
            cfg=cfg,
            clock_fn=fake_clock,
            breakers=breakers,
        )
        assert recovered_resp.provider == "groq_outage"
        assert recovered_resp.text == "Groq is restored!"
        assert cb_a.is_closed is True

    asyncio.run(run())


# ============================================================================
# Scenario 7: End-to-End FastAPI /chat Endpoint Outage Test
# ============================================================================
def test_scenario_7_api_endpoint_handles_outage_and_persists_attempts():
    """
    Scenario 7:
    End-to-End integration test through FastAPI POST /chat:
    - Primary provider fails (500).
    - Fallback provider answers with 200 OK.
    - Database RequestLog persists the attempt sequence with accurate JSON.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_primary"

    provider_b = MockProvider(failure_rate=0.0, reply="End-to-end fallback reply", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_secondary"

    test_router = Router(
        providers={"mock_primary": provider_a, "mock_secondary": provider_b},
        strategy=PriorityStrategy(priority=["mock_primary", "mock_secondary"]),
    )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_router] = lambda: test_router

    try:
        client = TestClient(app)
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "Outage scenario test"}]})

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "mock_secondary"
        assert data["text"] == "End-to-end fallback reply"

        # Verify SQLite request log
        db = TestingSessionLocal()
        log = db.query(RequestLog).order_by(RequestLog.id.desc()).first()
        assert log is not None
        assert log.status == "success"
        assert log.provider == "mock_secondary"
        assert isinstance(log.attempts, list)
        assert len(log.attempts) >= 2
        # First attempt was primary failure
        assert log.attempts[0]["provider"] == "mock_primary"
        assert log.attempts[0]["status"] == 502
        # Second attempt was secondary success
        assert log.attempts[-1]["provider"] == "mock_secondary"
        assert log.attempts[-1]["status"] == 200
        db.close()

    finally:
        app.dependency_overrides.clear()
