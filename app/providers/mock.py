import asyncio
import random
import time

from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


class MockProvider(BaseProvider):
    """Fake provider: answers with a canned reply after a random delay."""

    name = "mock"
    default_model = "mock-1"

    def __init__(
        self,
        min_delay: float = 0.2,
        max_delay: float = 1.0,
        reply: str = "This is a mock response.",
    ):
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError("need 0 <= min_delay <= max_delay")
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.reply = reply

    async def complete(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        # Non-blocking wait: the event loop keeps serving other requests.
        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))

        # Rough token estimate: one word = one token.
        tokens_in = sum(len(m.content.split()) for m in request.messages)
        tokens_out = len(self.reply.split())

        latency_ms = (time.perf_counter() - start) * 1000

        return ChatResponse(
            text=self.reply,
            provider=self.name,
            model=request.model or self.default_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )