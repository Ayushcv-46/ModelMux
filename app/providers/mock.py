import asyncio
import random
import time

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse

ERROR_KINDS = {
    "500": ProviderError,
    "timeout": ProviderTimeout,
    "429": ProviderRateLimited,
}


class MockProvider(BaseProvider):
    """Fake provider: canned reply after a random delay, with optional injected failures."""

    name = "mock"
    default_model = "mock-1"

    def __init__(
        self,
        min_delay: float = 0.2,
        max_delay: float = 1.0,
        reply: str = "This is a mock response.",
        failure_rate: float = 0.0,
        error_kind: str = "500",
        seed: int | None = None,
    ):
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError("need 0 <= min_delay <= max_delay")
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be between 0 and 1")
        if error_kind not in ERROR_KINDS:
            raise ValueError(f"error_kind must be one of {list(ERROR_KINDS)}")

        self.min_delay = min_delay
        self.max_delay = max_delay
        self.reply = reply
        self.failure_rate = failure_rate
        self.error_kind = error_kind
        self.rng = random.Random(seed)  # private generator: repeatable when seeded

    async def complete(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()

        # Non-blocking wait: the event loop keeps serving other requests.
        await asyncio.sleep(self.rng.uniform(self.min_delay, self.max_delay))

        # Fault injection: random() is in [0, 1), so rate 0 never fails, rate 1 always fails.
        if self.rng.random() < self.failure_rate:
            raise ERROR_KINDS[self.error_kind](f"mock {self.error_kind} failure")

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