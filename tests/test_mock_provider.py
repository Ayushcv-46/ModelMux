import asyncio
import time

from app.providers.mock import MockProvider
from app.schemas import ChatRequest, ChatResponse, Message


def make_request(text="hello there"):
    return ChatRequest(messages=[Message(role="user", content=text)])


async def test_returns_valid_chat_response():
    provider = MockProvider(min_delay=0.01, max_delay=0.02)
    response = await provider.complete(make_request())
    assert isinstance(response, ChatResponse)
    assert response.provider == "mock"
    assert response.model == "mock-1"


async def test_latency_falls_inside_range():
    provider = MockProvider(min_delay=0.05, max_delay=0.15)
    response = await provider.complete(make_request())
    # small tolerance: timers can fire slightly early, and scheduling adds overhead
    assert 45 <= response.latency_ms <= 250


async def test_tokens_estimated_as_word_count():
    provider = MockProvider(min_delay=0, max_delay=0, reply="one two three")
    response = await provider.complete(make_request("hello there"))
    assert response.tokens_in == 2
    assert response.tokens_out == 3


async def test_calls_run_concurrently_not_one_by_one():
    provider = MockProvider(min_delay=0.2, max_delay=0.2)
    start = time.perf_counter()
    await asyncio.gather(*(provider.complete(make_request()) for _ in range(10)))
    elapsed = time.perf_counter() - start
    # sequential would take ~2.0s; concurrent should take ~0.2s
    assert elapsed < 1.0