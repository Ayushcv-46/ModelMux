"""
Day 17: Contract Testing Across All Providers

Validates two fundamental behavioral contracts:
1. Happy Path Contract:
   Given any valid ChatRequest, every provider must return a valid ChatResponse
   with standardized types and non-negative metric values.
2. Error Handling Contract:
   On network connection failure, every provider must translate raw network errors
   into ModelMux's standard ProviderError (preserving the provider tag).
"""

import asyncio
import json

import httpx
import pytest

from app.errors import ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.ollama import OllamaProvider
from app.schemas import ChatRequest, Message

# ============================================================================
# 1. Mock Raw API Responses
# Each provider has a unique upstream response payload shape.
# ============================================================================

GROQ_MOCK_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello from Groq!",
            }
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    },
    "model": "openai/gpt-oss-20b",
}

GEMINI_MOCK_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello from Gemini!"}],
                "role": "model",
            }
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 4,
    },
}

OLLAMA_MOCK_RESPONSE = {
    "model": "qwen2.5:3b",
    "message": {
        "role": "assistant",
        "content": "Hello from Ollama!",
    },
    "prompt_eval_count": 7,
    "eval_count": 3,
}

# Standard input request used across all tests
SAMPLE_REQUEST = ChatRequest(
    messages=[Message(role="user", content="Say hello.")],
    temperature=0.7,
    max_tokens=50,
)


# ============================================================================
# CONTRACT 1: Happy Path Behavior
# ============================================================================

@pytest.mark.parametrize(
    "provider_name",
    ["groq", "gemini", "ollama"],
)
def test_all_providers_return_valid_chat_response(provider_name):
    """
    CONTRACT: Every provider.complete(ChatRequest) must return a ChatResponse
    matching the ModelMux schema contract.
    """
    mock_payloads = {
        "groq": GROQ_MOCK_RESPONSE,
        "gemini": GEMINI_MOCK_RESPONSE,
        "ollama": OLLAMA_MOCK_RESPONSE,
    }
    mock_body = json.dumps(mock_payloads[provider_name]).encode("utf-8")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=mock_body,
        )

    async def run():
        # Using async with ensures clean client lifecycle within the test loop
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
            if provider_name == "groq":
                provider = GroqProvider(api_key="mock-groq-key", client=client)
            elif provider_name == "gemini":
                provider = GeminiProvider(api_key="mock-gemini-key", client=client)
            elif provider_name == "ollama":
                provider = OllamaProvider(client=client)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

            return await provider.complete(SAMPLE_REQUEST)

    response = asyncio.run(run())

    # --- Verify the Contract Rules ---
    # 1. Output text must be a string
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    # 2. Provider identifier must match the provider being queried
    assert response.provider == provider_name

    # 3. Model identifier must be populated
    assert isinstance(response.model, str)
    assert len(response.model) > 0

    # 4. Token metrics must be non-negative numbers
    assert response.tokens_in >= 0
    assert response.tokens_out >= 0

    # 5. Latency must be non-negative
    assert response.latency_ms >= 0


# ============================================================================
# CONTRACT 2: Error Handling on Network Failure
# ============================================================================

@pytest.mark.parametrize(
    "provider_name",
    ["groq", "gemini", "ollama"],
)
def test_all_providers_raise_provider_error_on_connection_failure(provider_name):
    """
    CONTRACT: When network connection fails (httpx.ConnectError), every provider
    MUST catch it and re-raise ModelMux's standard ProviderError.
    """
    def error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by target host")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(error_handler)) as client:
            if provider_name == "groq":
                provider = GroqProvider(api_key="mock-groq-key", client=client)
            elif provider_name == "gemini":
                provider = GeminiProvider(api_key="mock-gemini-key", client=client)
            elif provider_name == "ollama":
                provider = OllamaProvider(client=client)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

            await provider.complete(SAMPLE_REQUEST)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run())

    # Verify that the standard ProviderError tagged the right provider
    assert exc_info.value.provider == provider_name
