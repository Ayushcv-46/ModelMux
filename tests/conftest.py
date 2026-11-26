"""
Pytest configuration and global fixtures for ModelMux.
"""

import pytest

from app.auth import get_current_api_key
from app.db import ApiKey
from app.main import app
from app.reliability.circuit_breaker import reset_circuit_breakers


@pytest.fixture(autouse=True)
def clean_circuit_breakers():
    """Ensure clean circuit breaker and in-memory rate limiter state before and after each test."""
    reset_circuit_breakers()
    if hasattr(app.state, "in_memory_limiter") and app.state.in_memory_limiter is not None:
        app.state.in_memory_limiter.clear()
    yield
    reset_circuit_breakers()
    if hasattr(app.state, "in_memory_limiter") and app.state.in_memory_limiter is not None:
        app.state.in_memory_limiter.clear()


@pytest.fixture(autouse=True)
def legacy_tests_auth_override(request):
    """
    Ensure backward compatibility for legacy tests (Days 1-30) that call /chat
    without an Authorization header, while letting auth tests (test_auth.py)
    test real authentication directly.
    """
    node_name = request.node.name
    file_name = request.node.fspath.basename if hasattr(request.node, "fspath") else ""

    # Do NOT override when running tests specifically designed to test authentication or per-key rate limits
    if "test_auth" in file_name or "test_api_keys" in file_name or "test_rate_limit_middleware" in file_name or "auth" in node_name:
        yield
        return

    dummy_key = ApiKey(
        id=999,
        name="legacy-test-key",
        key_hash="dummy_hash",
        rate_limit=1000,
        active=True,
        tenant_id="legacy-test-tenant",
    )
    app.dependency_overrides.setdefault(get_current_api_key, lambda: dummy_key)
    try:
        yield
    finally:
        if app.dependency_overrides.get(get_current_api_key) == (lambda: dummy_key) or app.dependency_overrides.get(get_current_api_key):
            app.dependency_overrides.pop(get_current_api_key, None)
