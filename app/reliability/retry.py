"""
Retry mechanism for ModelMux.
Implements bounded async retries with exponential backoff, randomized jitter,
intelligent error filtering, and total time budget enforcement.

Engineering Refinement (Idempotency, Duplicate Charges & Retry Budget):
------------------------------------------------------------------------
- Pre-generation failures: e.g. connect timeout, DNS failure, 503 before headers.
  The upstream LLM provider never processed the prompt. Retrying is completely safe
  and incurs zero financial or token cost.
- Post-generation / In-flight failures: e.g. read timeout while streaming or waiting
  for completion body after generation started.
  The upstream provider may have generated the tokens and billed the API account.
  Retrying sends the prompt again, which may result in duplicate token charges.
- Single Deadline Enforcement: A request never accumulates timeouts unboundedly.
  total_timeout_seconds sets a global hard deadline across all attempts, ensuring
  each successive attempt and backoff sleeps within the remaining budget.
"""

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

from app.errors import ProviderTimeout, get_retry_after, is_retryable

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
    total_timeout_seconds: float | None = None,
    sleep_fn: Callable[[float], Awaitable[Any]] | None = None,
    rng: random.Random | None = None,
    clock_fn: Callable[[], float] = time.perf_counter,
    on_attempt: Callable[..., Any] | None = None,
) -> T:
    """
    Execute an async zero-argument callable `fn` up to `attempts` times with
    exponential backoff, randomized jitter, total time budget enforcement,
    and granular attempt recording via on_attempt callback.

    Guarantees:
    - Retries only transient / retryable errors (as classified by is_retryable).
    - Stops immediately on non-retryable client errors (400, 401, etc.).
    - Backoff grows exponentially with attempts, capped at max_delay.
    - Total wall-clock time is bounded by total_timeout_seconds.
    - Attempts and backoff sleeps stop immediately when budget is exhausted.
    - Sleep function and clock function are injectable for fast, deterministic unit testing.
    - Invokes on_attempt(status=..., duration_ms=..., error=...) after each attempt.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    sleeper = sleep_fn if sleep_fn is not None else asyncio.sleep
    last_error: Exception | None = None
    start_time = clock_fn()

    for attempt in range(1, attempts + 1):
        # 1. Check if total time budget is already exhausted before starting this attempt
        if total_timeout_seconds is not None:
            elapsed = clock_fn() - start_time
            remaining = total_timeout_seconds - elapsed
            if remaining <= 0:
                logger.warning(
                    f"Retry budget of {total_timeout_seconds:.2f}s exhausted before attempt {attempt} (elapsed: {elapsed:.2f}s)."
                )
                if last_error is not None:
                    raise last_error
                raise ProviderTimeout(
                    f"Total request budget of {total_timeout_seconds:.2f}s exceeded before attempt {attempt}.",
                    provider="router",
                )

        attempt_start = clock_fn()
        try:
            # 2. Enforce one deadline: bind this attempt by the remaining budget
            if total_timeout_seconds is not None:
                elapsed = clock_fn() - start_time
                remaining = total_timeout_seconds - elapsed
                if remaining <= 0:
                    raise TimeoutError(f"Budget of {total_timeout_seconds:.2f}s exhausted")
                async with asyncio.timeout(remaining):
                    result = await fn()
            else:
                result = await fn()

            duration_ms = int(round((clock_fn() - attempt_start) * 1000))
            if on_attempt is not None:
                on_attempt(status=200, duration_ms=duration_ms, error=None)
            return result

        except Exception as err:
            duration_ms = int(round((clock_fn() - attempt_start) * 1000))

            # If the budget timeout was triggered, wrap as ProviderTimeout
            if isinstance(err, TimeoutError) and total_timeout_seconds is not None:
                elapsed = clock_fn() - start_time
                err = ProviderTimeout(
                    f"Total request budget of {total_timeout_seconds:.2f}s exceeded after {elapsed:.2f}s.",
                    provider="router",
                )

            status_code = getattr(err, "status_code", 500)
            if on_attempt is not None:
                on_attempt(status=status_code, duration_ms=duration_ms, error=err)

            last_error = err

            # 3. Stop immediately on non-retryable errors (e.g. 400, 401, bad model)
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

            # Check if upstream requested Retry-After
            retry_after = get_retry_after(err)
            if retry_after is not None and retry_after > delay:
                delay = retry_after

            # 4. Check if backoff delay would exceed the remaining time budget
            if total_timeout_seconds is not None:
                elapsed = clock_fn() - start_time
                remaining = total_timeout_seconds - elapsed
                if delay >= remaining:
                    logger.warning(
                        f"Backoff delay of {delay:.2f}s exceeds remaining budget {remaining:.2f}s. Stopping retries."
                    )
                    break

            logger.info(
                f"Attempt {attempt}/{attempts} failed with retryable error: {err}. Retrying in {delay:.2f}s..."
            )

            if delay > 0:
                await sleeper(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected state: retry loop completed without returning or raising.")
