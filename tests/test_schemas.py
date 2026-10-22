import pytest
from pydantic import ValidationError

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