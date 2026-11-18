import asyncio
import json
import httpx
import pytest

from app.errors import ProviderError, ProviderTimeout
from app.providers.ollama import OllamaProvider
from app.schemas import ChatRequest, ChatResponse, Message


MOCK_OLLAMA_RESPONSE = {
    "model": "qwen2.5:3b",
    "created_at": "2026-11-17T01:00:00Z",
    "message": {
        "role": "assistant",
        "content": "Hello!",
    },
    "done": True,
    "total_duration": 500000000,
    "prompt_eval_count": 8,
    "eval_count": 2,
}


@pytest.fixture
def sample_request() -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Say hello in one word")],
        model="qwen2.5:3b",
        temperature=0.7,
        max_tokens=50,
    )


def test_ollama_translates_request_and_response_correctly(sample_request):
    """Test that ChatRequest translates to Ollama payload and response maps to ChatResponse."""
    recorded_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recorded_request
        recorded_request["url"] = str(request.url)
        recorded_request["method"] = request.method
        recorded_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=MOCK_OLLAMA_RESPONSE)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(client=client)
            return await provider.complete(sample_request)

    response = asyncio.run(run())

    # Verify HTTP request sent to Ollama
    assert recorded_request["url"] == "http://localhost:11434/api/chat"
    assert recorded_request["method"] == "POST"
    assert recorded_request["body"]["model"] == "qwen2.5:3b"
    assert recorded_request["body"]["stream"] is False
    assert recorded_request["body"]["messages"] == [{"role": "user", "content": "Say hello in one word"}]
    assert recorded_request["body"]["options"]["temperature"] == 0.7
    assert recorded_request["body"]["options"]["num_predict"] == 50

    # Verify ChatResponse normalized output
    assert isinstance(response, ChatResponse)
    assert response.provider == "ollama"
    assert response.model == "qwen2.5:3b"
    assert response.text == "Hello!"
    assert response.tokens_in == 8
    assert response.tokens_out == 2
    assert response.latency_ms >= 0


def test_ollama_uses_default_model_when_none_provided():
    """Verify fallback to default_model when request.model is None."""
    req = ChatRequest(messages=[Message(role="user", content="Hi")], model=None)

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content.decode("utf-8"))
        assert data["model"] == "qwen2.5:3b"
        return httpx.Response(200, json={"message": {"content": "Hi there"}, "prompt_eval_count": 2, "eval_count": 2})

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(default_model="qwen2.5:3b", client=client)
            return await provider.complete(req)

    resp = asyncio.run(run())
    assert resp.provider == "ollama"
    assert resp.text == "Hi there"


def test_ollama_connect_error_raises_provider_error(sample_request):
    """Test that network connection refusal maps to ProviderError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(client=client)
            await provider.complete(sample_request)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "ollama"
    assert "Failed to connect to Ollama" in exc_info.value.message


def test_ollama_timeout_raises_provider_timeout(sample_request):
    """Test that HTTP timeout maps to ProviderTimeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(client=client)
            await provider.complete(sample_request)

    with pytest.raises(ProviderTimeout) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "ollama"
    assert "timed out" in exc_info.value.message


def test_ollama_server_error_status_raises_provider_error(sample_request):
    """Test that HTTP 500 status from Ollama server raises ProviderError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal server error in llama.cpp")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(client=client)
            await provider.complete(sample_request)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.provider == "ollama"
    assert "Ollama returned HTTP 500" in exc_info.value.message
