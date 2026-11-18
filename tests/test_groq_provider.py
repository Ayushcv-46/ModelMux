import asyncio
import json
import httpx
import pytest

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.groq import GroqProvider
from app.schemas import ChatRequest, ChatResponse, Message


MOCK_GROQ_RESPONSE = {
    "id": "chatcmpl-12345",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "openai/gpt-oss-20b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from Groq!",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    },
}


@pytest.fixture
def sample_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello in one word!"),
        ],
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=32,
    )


def test_groq_translates_request_and_headers_correctly(sample_request):
    """Verify OpenAI-compatible payload, Bearer auth header, and ChatResponse extraction."""
    recorded_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recorded_request
        recorded_request["url"] = str(request.url)
        recorded_request["method"] = request.method
        recorded_request["headers"] = dict(request.headers)
        recorded_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=MOCK_GROQ_RESPONSE)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GroqProvider(api_key="gsk-mock-key-123", client=client)
            return await provider.complete(sample_request)

    response = asyncio.run(run())

    # 1. URL and headers check
    assert recorded_request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert recorded_request["method"] == "POST"
    assert recorded_request["headers"]["authorization"] == "Bearer gsk-mock-key-123"

    # 2. Payload structure check
    body = recorded_request["body"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello in one word!"},
    ]
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 32

    # 3. ChatResponse extraction check
    assert isinstance(response, ChatResponse)
    assert response.provider == "groq"
    assert response.model == "openai/gpt-oss-20b"
    assert response.text == "Hello from Groq!"
    assert response.tokens_in == 12
    assert response.tokens_out == 4
    assert response.latency_ms >= 0


def test_groq_rate_limited_returns_429():
    """Verify HTTP 429 maps to ProviderRateLimited."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limit reached: requests per minute")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GroqProvider(api_key="gsk-test", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderRateLimited) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "groq"
    assert exc_info.value.status_code == 429


def test_groq_server_error_returns_502():
    """Verify HTTP 500/503 maps to ProviderError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Groq internal server error")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GroqProvider(api_key="gsk-test", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "groq"
    assert exc_info.value.status_code == 502


def test_groq_timeout_returns_504():
    """Verify HTTP timeout maps to ProviderTimeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Groq request timed out")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GroqProvider(api_key="gsk-test", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderTimeout) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "groq"
    assert exc_info.value.status_code == 504


def test_groq_missing_api_key_raises_error():
    """Verify missing API key raises ProviderError immediately without network calls."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200))

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GroqProvider(api_key="", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "groq"
    assert "Groq API key is not configured" in exc_info.value.message
