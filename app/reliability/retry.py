"""
Retry mechanism for ModelMux.
Implements bounded async retries with exponential backoff, randomized jitter,
and intelligent error filtering.

Engineering Refinement (Idempotency, Duplicate Charges & Jitter):
------------------------------------------------------------------
- Pre-generation failures: e.g. connect timeout, DNS failure, 503 before headers.
  The upstream LLM provider never processed the prompt. Retrying is completely safe
  and incurs zero financial or token cost.
- Post-generation / In-flight failures: e.g. read timeout while streaming or waiting
  for completion body after generation started.
  The upstream provider may have generated the tokens and billed the API account.
  Retrying sends the prompt again, which may result in duplicate token charges.
- Randomized Capped Jitter: Decorrelates client retries across time, preventing
  the "Thundering Herd" / retry storms on upstream services.
"""

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, TypeVar

from app.errors import get_retry_after, is_retryable

logger = logging.getLogger(__name__)

T = TypeVar("T")


def calculate_backoff(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: bool = True,
    rng: random.Random | None = None,
) -> float:
    """
    Calculate exponential backoff delay with optional randomized capped jitter.

    Formula:
    raw_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    With Jitter:
    jittered_delay = uniform(0.5 * raw_delay, raw_delay)
    """
    if base_delay <= 0:
        return 0.0

    raw_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))

    if not jitter or raw_delay <= 0:
        return raw_delay

    generator = rng or random
    # Capped randomized jitter: ensures delays grow while preventing synchronized retries
    return generator.uniform(0.5 * raw_delay, raw_delay)


async def retry(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: bool = True,
    sleep_fn: Callable[[float], Awaitable[Any]] | None = None,
    rng: random.Random | None = None,
) -> T:
    """
    Execute an async zero-argument callable `fn` up to `attempts` times with
    exponential backoff and randomized jitter.

    Guarantees:
    - Retries only transient / retryable errors (as classified by is_retryable).
    - Stops immediately on non-retryable client errors (400, 401, etc.).
    - Backoff grows exponentially with attempts, capped at max_delay.
    - Sleep function is injectable for instantaneous unit testing.
    - Upstream Retry-After header/attribute is respected if present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    sleeper = sleep_fn if sleep_fn is not None else asyncio.sleep
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as err:
            last_error = err

            # Stop immediately on non-retryable errors (e.g. 400, 401, bad model)
            if not is_retryable(err):
                logger.warning(
                    f"Attempt {attempt}/{attempts} failed with non-retryable error: {err}. Aborting retries."
                )
                raise err

            # If all attempts are exhausted, break out to raise last error
            if attempt == attempts:
                logger.warning(
                    f"Attempt {attempt}/{attempts} failed with retryable error: {err}. Exhausted all attempts."
                )
                break

            # Calculate backoff delay
            delay = calculate_backoff(
                attempt=attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                rng=rng,
            )

            # If upstream provided a Retry-After constraint, ensure we wait at least that long
            retry_after = get_retry_after(err)
            if retry_after is not None and retry_after > delay:
                delay = retry_after

            logger.info(
                f"Attempt {attempt}/{attempts} failed with retryable error: {err}. Retrying in {delay:.2f}s..."
            )

            # Sleep using the injected or default async sleep function
            if delay > 0:
                await sleeper(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected state: retry loop completed without returning or raising.")
