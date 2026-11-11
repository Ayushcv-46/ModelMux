"""
Day 8 tests — POST /chat endpoint (full HTTP layer).

We use httpx.AsyncClient with ASGI transport, run via asyncio.run()
inside sync test functions — matching the pattern already used in
test_provider_interface.py and test_mock_provider.py.
"""

import asyncio

import httpx
import pytest

from app.main import app, get_provider
from app.providers.mock import MockProvider
from app.schemas import ChatResponse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BODY = {"messages": [{"role": "user", "content": "Hello!"}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transport(provider: MockProvider) -> httpx.ASGITransport:
    """Inject provider and return an ASGI transport pointed at the app."""
    app.dependency_overrides[get_provider] = lambda: provider
    return httpx.ASGITransport(app=app)


async def post_chat(provider: MockProvider, body: dict) -> httpx.Response:
    """Single async helper: open a client, POST /chat, return the response."""
    transport = make_transport(provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/chat", json=body)


def chat(provider: MockProvider, body: dict = VALID_BODY) -> httpx.Response:
    """Sync wrapper so each test stays a plain def."""
    return asyncio.run(post_chat(provider, body))


def fast(**kwargs) -> MockProvider:
    """Zero-delay MockProvider with optional overrides."""
    return MockProvider(min_delay=0, max_delay=0, **kwargs)


# ---------------------------------------------------------------------------
# Teardown fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_chat_returns_200():
    assert chat(fast()).status_code == 200


def test_chat_response_has_correct_keys():
    body = chat(fast()).json()
    assert set(body.keys()) == {"text", "provider", "model", "tokens_in", "tokens_out", "latency_ms"}


def test_chat_response_validates_as_chat_response():
    body = chat(fast()).json()
    parsed = ChatResponse(**body)
    assert parsed.provider == "mock"


def test_chat_uses_provider_reply():
    body = chat(fast(reply="Custom reply")).json()
    assert body["text"] == "Custom reply"


def test_chat_respects_model_field():
    response = chat(fast(), body={**VALID_BODY, "model": "gpt-99"})
    assert response.json()["model"] == "gpt-99"


def test_chat_uses_default_model_when_none_given():
    assert chat(fast()).json()["model"] == "mock-1"


def test_tokens_in_matches_word_count():
    body = {"messages": [{"role": "user", "content": "one two three"}]}
    response = chat(fast(reply="x"), body=body)
    assert response.json()["tokens_in"] == 3


def test_latency_ms_is_non_negative():
    assert chat(fast()).json()["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Validation-error tests (Pydantic rejects bad input → 422)
# ---------------------------------------------------------------------------


def test_empty_messages_list_returns_422():
    assert chat(fast(), body={"messages": []}).status_code == 422


def test_missing_messages_returns_422():
    assert chat(fast(), body={}).status_code == 422


def test_bad_temperature_returns_422():
    assert chat(fast(), body={**VALID_BODY, "temperature": 99}).status_code == 422


def test_bad_max_tokens_returns_422():
    assert chat(fast(), body={**VALID_BODY, "max_tokens": 0}).status_code == 422


# ---------------------------------------------------------------------------
# Error-handling tests (ProviderError → structured JSON error response)
# ---------------------------------------------------------------------------


def test_provider_timeout_returns_504():
    response = chat(fast(failure_rate=1.0, error_kind="timeout"))
    assert response.status_code == 504
    assert response.json()["error"]["type"] == "ProviderTimeout"


def test_provider_rate_limited_returns_429():
    response = chat(fast(failure_rate=1.0, error_kind="429"))
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "ProviderRateLimited"


def test_provider_generic_error_returns_502():
    response = chat(fast(failure_rate=1.0, error_kind="500"))
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "ProviderError"


def test_error_body_has_correct_keys():
    body = chat(fast(failure_rate=1.0, error_kind="timeout")).json()
    assert set(body["error"].keys()) == {"type", "message", "provider"}


def test_error_body_provider_key_exists():
    body = chat(fast(failure_rate=1.0, error_kind="timeout")).json()
    assert "provider" in body["error"]
