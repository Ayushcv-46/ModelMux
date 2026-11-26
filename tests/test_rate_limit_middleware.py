"""
Day 36 — Rate Limiter Middleware / Endpoint Integration Tests

Verifies:
1. Limiter runs right after auth using per-key `rate_limit` from `ApiKey`.
2. Successful responses include `X-RateLimit-Remaining` header decreasing per call.
3. Exceeding a key's limit returns HTTP 429 with `Retry-After` and `X-RateLimit-Remaining: 0`.
4. Two keys with different limits are isolated: throttling key A does not affect key B.
5. Works seamlessly with Redis-backed `RateLimiter` (via FakeAsyncRedis).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import create_api_key
from app.db import Base, get_db, init_db
from app.main import app, get_provider, get_rate_limiter
from app.providers.mock import MockProvider
from app.ratelimit import RateLimiter
from tests.test_redis_limiter import FakeAsyncRedis


@pytest.fixture
def rate_limit_db():
    """Isolated in-memory SQLite DB + MockProvider for Day 36 tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_provider] = lambda: MockProvider(min_delay=0.0, max_delay=0.0)

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_provider, None)
    app.dependency_overrides.pop(get_rate_limiter, None)


def test_two_keys_different_limits_one_throttled_other_succeeds(rate_limit_db):
    """
    Day 36 Core Goal:
    Key A has rate_limit=2, Key B has rate_limit=10.
    Sending 3 requests on Key A throttles the 3rd with 429 + Retry-After + X-RateLimit-Remaining=0,
    while Key B continues succeeding with 200 OK.
    """
    _, key_a = create_api_key(rate_limit_db, name="free-tier", rate_limit=2, tenant_id="tenant-free")
    _, key_b = create_api_key(rate_limit_db, name="pro-tier", rate_limit=10, tenant_id="tenant-pro")

    fake_redis = FakeAsyncRedis()
    limiter = RateLimiter(fake_redis)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    client = TestClient(app)
    payload = {"messages": [{"role": "user", "content": "Hello"}]}

    # Key A: 1st request -> 200 OK, remaining=1
    r1 = client.post("/v1/chat", headers={"Authorization": f"Bearer {key_a}"}, json=payload)
    assert r1.status_code == 200
    assert r1.headers["X-RateLimit-Remaining"] == "1"

    # Key A: 2nd request -> 200 OK, remaining=0
    r2 = client.post("/v1/chat", headers={"Authorization": f"Bearer {key_a}"}, json=payload)
    assert r2.status_code == 200
    assert r2.headers["X-RateLimit-Remaining"] == "0"

    # Key A: 3rd request -> 429 Too Many Requests!
    r3 = client.post("/v1/chat", headers={"Authorization": f"Bearer {key_a}"}, json=payload)
    assert r3.status_code == 429
    assert int(r3.headers["Retry-After"]) >= 1
    assert r3.headers["X-RateLimit-Remaining"] == "0"

    # Key B: still has its full 10-token limit -> 200 OK, remaining=9
    rb = client.post("/v1/chat", headers={"Authorization": f"Bearer {key_b}"}, json=payload)
    assert rb.status_code == 200
    assert rb.headers["X-RateLimit-Remaining"] == "9"
