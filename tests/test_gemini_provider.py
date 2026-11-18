import asyncio
import json
import httpx
import pytest

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.gemini import GeminiProvider
from app.schemas import ChatRequest, ChatResponse, Message


MOCK_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello from Gemini!"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    },
}


@pytest.fixture
def sample_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi! How can I help?"),
            Message(role="user", content="Tell me a joke."),
        ],
        model="gemini-3.5-flash-lite",
        temperature=0.8,
        max_tokens=64,
    )


def test_gemini_translates_request_and_roles_correctly(sample_request):
    """Verify role mapping (assistant->model, system->systemInstruction) and payload structure."""
    recorded_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recorded_request
        recorded_request["url"] = str(request.url)
        recorded_request["method"] = request.method
        recorded_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=MOCK_GEMINI_RESPONSE)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GeminiProvider(api_key="test-key-123", client=client)
            return await provider.complete(sample_request)

    response = asyncio.run(run())

    # 1. URL and query param check
    assert "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent" in recorded_request["url"]
    assert "key=test-key-123" in recorded_request["url"]
    assert recorded_request["method"] == "POST"

    # 2. Role mappings check
    body = recorded_request["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "You are a helpful assistant."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello!"}]},
        {"role": "model", "parts": [{"text": "Hi! How can I help?"}]},
        {"role": "user", "parts": [{"text": "Tell me a joke."}]},
    ]
    assert body["generationConfig"]["temperature"] == 0.8
    assert body["generationConfig"]["maxOutputTokens"] == 64

    # 3. ChatResponse fields check
    assert isinstance(response, ChatResponse)
    assert response.provider == "gemini"
    assert response.model == "gemini-3.5-flash-lite"
    assert response.text == "Hello from Gemini!"
    assert response.tokens_in == 10
    assert response.tokens_out == 5
    assert response.latency_ms >= 0


def test_gemini_rate_limited_returns_429():
    """Verify HTTP 429 maps to ProviderRateLimited."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Resource exhausted: quota exceeded")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GeminiProvider(api_key="test-key", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderRateLimited) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.status_code == 429


def test_gemini_server_error_returns_502():
    """Verify HTTP 500/503 maps to ProviderError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GeminiProvider(api_key="test-key", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.status_code == 502


def test_gemini_timeout_returns_504():
    """Verify HTTP timeout maps to ProviderTimeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GeminiProvider(api_key="test-key", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderTimeout) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.status_code == 504


def test_missing_api_key_raises_error():
    """Verify missing API key raises ProviderError immediately without calling network."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200))

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            provider = GeminiProvider(api_key="", client=client)
            req = ChatRequest(messages=[Message(role="user", content="hi")])
            await provider.complete(req)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "gemini"
    assert "API key is not configured" in exc_info.value.message
