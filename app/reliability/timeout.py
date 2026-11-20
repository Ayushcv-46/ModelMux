"""
Timeout execution wrapper using asyncio.timeout.
Guarantees:
- Every provider call is strictly bounded by a deadline.
- When a timeout occurs, background network tasks are cancelled immediately.
- Raises ProviderTimeout with elapsed wall-clock duration.
"""

import asyncio
import time

from app.errors import ProviderTimeout
from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


async def execute_with_timeout(
    provider: BaseProvider,
    request: ChatRequest,
    timeout_seconds: float,
) -> ChatResponse:
    """
    Execute provider.complete(request) bounded by timeout_seconds.
    If the timeout expires, the operation is cancelled and ProviderTimeout is raised.
    """
    start = time.perf_counter()
    provider_name = getattr(provider, "name", "unknown")

    try:
        async with asyncio.timeout(timeout_seconds):
            return await provider.complete(request)
    except TimeoutError as exc:
        elapsed = time.perf_counter() - start
        raise ProviderTimeout(
            message=f"Provider '{provider_name}' timed out after {elapsed:.2f}s (configured limit: {timeout_seconds:.2f}s)",
            provider=provider_name,
        ) from exc
