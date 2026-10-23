import pytest
from pydantic import ValidationError
from app.schemas import ChatRequest, ChatResponse
from app.schemas import ChatRequest


def test_valid_request_fills_defaults():
    req = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
    assert req.messages[0].content == "Hello"
    assert req.model is None
    assert req.max_tokens == 256
    assert req.temperature == 1.0


def test_missing_messages_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ChatRequest()
    assert "messages" in str(exc.value)


def test_bad_temperature_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=5,
        )
    assert "temperature" in str(exc.value)


def test_valid_response():
    resp = ChatResponse(
        text="Hi",
        provider="mock",
        model="mock-1",
        tokens_in=3,
        tokens_out=5,
        latency_ms=12.5,
    )
    assert resp.provider == "mock"
    assert resp.tokens_out == 5


def test_negative_tokens_rejected():
    with pytest.raises(ValidationError) as exc:
        ChatResponse(
            text="Hi",
            provider="mock",
            model="mock-1",
            tokens_in=-1,
            tokens_out=5,
            latency_ms=12.5,
        )
    assert "tokens_in" in str(exc.value)