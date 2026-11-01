import asyncio

import pytest

from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


class EchoProvider(BaseProvider):
    name = "echo"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            text=request.messages[-1].content,
            provider=self.name,
            model="echo-1",
            tokens_in=1,
            tokens_out=1,
            latency_ms=0,
        )


class NoCompleteProvider(BaseProvider):
    name = "broken"


def test_echo_provider_returns_last_message():
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )
    response = asyncio.run(EchoProvider().complete(request))
    assert response.text == "second"
    assert response.provider == "echo"


def test_provider_without_complete_is_rejected():
    with pytest.raises(TypeError):
        NoCompleteProvider()


def test_base_provider_cannot_be_used_directly():
    with pytest.raises(TypeError):
        BaseProvider()


def test_echo_provider_is_a_base_provider():
    assert isinstance(EchoProvider(), BaseProvider)