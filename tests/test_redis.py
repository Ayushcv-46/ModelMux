"""
Day 33 — Redis in Docker & Health Check Unit Tests

Verifies:
1. /health reports status="ok", redis="up" when Redis responds to PING.
2. /health reports status="degraded", redis="down" when Redis is unreachable or throws.
3. /health reports status="degraded", redis="down" when app.state.redis is None.
4. get_redis dependency returns active client or None.
5. Lifespan context manager gracefully handles Redis connection failures without crashing.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_redis, lifespan


class MockRedisHealthy:
    """Simulates a healthy, connected Redis server."""
    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


class MockRedisFailing:
    """Simulates a Redis server that dropped connection."""
    async def ping(self):
        raise ConnectionError("Connection refused by Redis")

    async def aclose(self) -> None:
        pass


def test_health_reports_redis_up():
    """When Redis is reachable, GET /health returns status=ok and redis=up."""
    original = getattr(app.state, "redis", None)
    app.state.redis = MockRedisHealthy()
    try:
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "redis": "up"}
    finally:
        app.state.redis = original


def test_health_reports_redis_down_on_connection_error():
    """When Redis throws ConnectionError on ping, GET /health returns degraded and redis=down."""
    original = getattr(app.state, "redis", None)
    app.state.redis = MockRedisFailing()
    try:
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "degraded", "redis": "down"}
    finally:
        app.state.redis = original


def test_health_reports_redis_down_when_redis_client_none():
    """When app.state.redis is None, GET /health returns degraded and redis=down."""
    original = getattr(app.state, "redis", None)
    app.state.redis = None
    try:
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "degraded", "redis": "down"}
    finally:
        app.state.redis = original


def test_get_redis_dependency():
    """get_redis returns the redis client attached to app.state."""
    mock_redis = MockRedisHealthy()
    original = getattr(app.state, "redis", None)
    app.state.redis = mock_redis
    try:
        class DummyRequest:
            app = type("App", (), {"state": app.state})()

        assert get_redis(DummyRequest()) is mock_redis
    finally:
        app.state.redis = original


@pytest.mark.asyncio
async def test_lifespan_degraded_fallback_when_redis_unavailable():
    """Lifespan should not crash when Redis server is not running on localhost."""
    test_app = type("TestApp", (), {"state": type("State", (), {})()})()
    async with lifespan(test_app):
        # Even if Redis is not running locally, app boots up safely in degraded mode
        assert hasattr(test_app.state, "http_client")
        assert hasattr(test_app.state, "redis")
