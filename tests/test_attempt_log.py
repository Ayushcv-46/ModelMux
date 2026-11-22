"""
Day 27 — Attempt Log Unit Tests

Validates:
1. Two-provider failure: Provider A fails (500), Provider B succeeds (200) -> both attempts recorded in order.
2. Retry + Fallback attempt sequence: Provider A retries and exhausts, Provider B succeeds -> all attempts logged sequentially.
3. All-providers-failed scenario: Both candidate providers fail -> full failure trail preserved.
4. Security & Sanitization: Raw API keys, Bearer tokens, and secrets are sanitized and never logged.
5. Database Persistence: SQLite JSON column correctly serializes and stores attempt logs.
6. History API Exposure: GET /v1/requests returns the attempts list with status, duration, provider, and error category.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import AppConfig, ProviderConfig, ReliabilityConfig
from app.db import Base, RequestLog, get_db, init_db
from app.errors import AllProvidersFailed, ProviderError
from app.main import app, get_provider, get_requests
from app.providers.mock import MockProvider
from app.reliability.fallback import execute_fallback_chain
from app.reliability.sanitizer import sanitize_error, sanitize_error_message
from app.schemas import ChatRequest, Message, RequestLogResponse


def make_request(prompt: str = "Hello", model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content=prompt)],
        model=model,
    )


# ---------------------------------------------------------------------------
# Test 1: Two-provider failure (A fails -> B succeeds) shows both attempts
# ---------------------------------------------------------------------------
def test_two_provider_failure_records_both_attempts():
    """
    Day 27 Core Goal:
    Provider A fails (500), fallback switches to Provider B (200).
    Verify that attempts list records both attempts with provider, status, duration, and error.
    """
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = MockProvider(failure_rate=0.0, reply="Provider B response", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=1, base_delay=0.0, total_timeout_seconds=5.0),
        providers={
            "mock_a": ProviderConfig(timeout_seconds=2.0),
            "mock_b": ProviderConfig(timeout_seconds=2.0),
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
        assert response.provider == "mock_b"

    asyncio.run(run())

    # Assert exactly two attempts recorded
    assert len(attempts) == 2

    # Attempt 1: mock_a failed (ProviderError -> 502)
    att1 = attempts[0]
    assert att1["attempt"] == 1
    assert att1["provider"] == "mock_a"
    assert att1["status"] == 502
    assert att1["error_category"] == "ProviderError"
    assert att1["error"] is not None
    assert att1["duration_ms"] >= 0

    # Attempt 2: mock_b succeeded
    att2 = attempts[1]
    assert att2["attempt"] == 2
    assert att2["provider"] == "mock_b"
    assert att2["status"] == 200
    assert att2["error"] is None
    assert att2["error_category"] is None
    assert att2["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# Test 2: Retry + Fallback logs every attempt in monotonic order
# ---------------------------------------------------------------------------
def test_retry_and_fallback_records_monotonic_attempts():
    """
    Provider A fails 2 times (1 initial attempt + 1 retry).
    Fallback switches to Provider B, which succeeds on attempt 1.
    Total attempts logged must be 3, with attempt numbers 1, 2, 3.
    """
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = MockProvider(failure_rate=0.0, reply="Success from B", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=2, base_delay=0.0, total_timeout_seconds=5.0),
        providers={
            "mock_a": ProviderConfig(timeout_seconds=2.0),
            "mock_b": ProviderConfig(timeout_seconds=2.0),
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
        assert response.provider == "mock_b"

    asyncio.run(run())

    assert len(attempts) == 3
    assert [a["attempt"] for a in attempts] == [1, 2, 3]

    # Attempts 1 and 2 belong to mock_a (failed with ProviderError -> 502)
    assert attempts[0]["provider"] == "mock_a"
    assert attempts[0]["status"] == 502
    assert attempts[1]["provider"] == "mock_a"
    assert attempts[1]["status"] == 502

    # Attempt 3 belongs to mock_b
    assert attempts[2]["provider"] == "mock_b"
    assert attempts[2]["status"] == 200
    assert attempts[2]["error"] is None


# ---------------------------------------------------------------------------
# Test 3: All providers fail -> full audit trail of failures preserved
# ---------------------------------------------------------------------------
def test_all_providers_failed_preserves_attempts():
    """When all candidates fail, attempts log retains every attempt up to the 503."""
    provider_a = MockProvider(failure_rate=1.0, error_kind="500", min_delay=0.0, max_delay=0.0)
    provider_a.name = "mock_a"

    provider_b = MockProvider(failure_rate=1.0, error_kind="timeout", min_delay=0.0, max_delay=0.0)
    provider_b.name = "mock_b"

    cfg = AppConfig(
        reliability=ReliabilityConfig(max_attempts=1, base_delay=0.0, total_timeout_seconds=5.0),
        providers={
            "mock_a": ProviderConfig(timeout_seconds=2.0),
            "mock_b": ProviderConfig(timeout_seconds=2.0),
        },
    )

    attempts: list[dict] = []

    async def run():
        with pytest.raises(AllProvidersFailed):
            await execute_fallback_chain(
                providers=[provider_a, provider_b],
                request=make_request(),
                cfg=cfg,
                attempts_log=attempts,
            )

    asyncio.run(run())

    assert len(attempts) == 2
    assert attempts[0]["provider"] == "mock_a"
    assert attempts[0]["status"] == 502
    assert attempts[1]["provider"] == "mock_b"
    assert attempts[1]["status"] == 504


# ---------------------------------------------------------------------------
# Test 4: Credential sanitization ensures raw secrets are never logged
# ---------------------------------------------------------------------------
def test_credential_sanitization():
    """Verify that sanitize_error scrubs Bearer tokens, OpenAI/Groq keys, and URI secret params."""
    raw_message = (
        "Upstream auth error: Bearer gsk_abc1234567890xyz with key=super_secret_token_123 "
        "and sk-proj-1234567890abcdef and AIzaSyD1234567890123456789012345678901"
    )

    sanitized = sanitize_error_message(raw_message)

    # Check that secrets are completely gone
    assert "gsk_abc1234567890xyz" not in sanitized
    assert "super_secret_token_123" not in sanitized
    assert "sk-proj-1234567890abcdef" not in sanitized
    assert "AIzaSyD1234567890123456789012345678901" not in sanitized

    # Check redaction placeholders are present
    assert "Bearer [REDACTED]" in sanitized
    assert "key=[REDACTED]" in sanitized
    assert "[REDACTED_API_KEY]" in sanitized

    # Check sanitize_error wrapper includes exception category
    err = ProviderError(raw_message)
    formatted = sanitize_error(err)
    assert formatted.startswith("ProviderError: ")
    assert "super_secret_token_123" not in formatted


# ---------------------------------------------------------------------------
# Test 5 & 6: Database Persistence & History API exposure
# ---------------------------------------------------------------------------
@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database populated via FastAPI calls."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield session

    session.close()
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_provider, None)


def test_chat_persists_attempts_and_exposes_via_requests_api(test_db_session):
    """
    Integration test:
    1. Send a request to /chat.
    2. Read from test database to verify attempts JSON column is populated.
    3. Query GET /v1/requests and verify attempts list is exposed in API response.
    """
    client = TestClient(app)

    # Inject mock provider for chat
    app.dependency_overrides[get_provider] = lambda: MockProvider(min_delay=0.0, max_delay=0.0)

    # 1. Successful POST /chat
    res = client.post("/chat", json={"messages": [{"role": "user", "content": "Explain attempts"}]})
    assert res.status_code == 200

    # 2. Check DB row
    rows = test_db_session.query(RequestLog).all()
    assert len(rows) == 1
    assert rows[0].attempts is not None
    assert len(rows[0].attempts) == 1
    assert rows[0].attempts[0]["provider"] == "mock"
    assert rows[0].attempts[0]["status"] == 200
    assert rows[0].attempts[0]["attempt"] == 1

    # 3. Query GET /v1/requests
    api_res = client.get("/v1/requests")
    assert api_res.status_code == 200
    items = api_res.json()
    assert len(items) == 1
    assert "attempts" in items[0]
    assert items[0]["attempts"] == rows[0].attempts
    assert items[0]["attempts"][0]["status"] == 200
