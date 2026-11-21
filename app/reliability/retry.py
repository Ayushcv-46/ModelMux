"""
Retry mechanism for ModelMux.
Implements bounded async retries with intelligent error filtering.

Engineering Refinement (Idempotency & Duplicate Charges):
---------------------------------------------------------
- Pre-generation failures: e.g. connect timeout, DNS failure, 503 before headers.
  The upstream LLM provider never processed the prompt. Retrying is completely safe
  and incurs zero financial or token cost.
- Post-generation / In-flight failures: e.g. read timeout while streaming or waiting
  for completion body after generation started.
  The upstream provider may have generated the tokens and billed the API account.
  Retrying sends the prompt again, which may result in duplicate token charges.
  Until upstream providers support end-to-end idempotency keys, this represents an
  inherent distributed systems trade-off between availability and cost.
"""

import logging
from typing import Awaitable, Callable, TypeVar

from app.errors import is_retryable

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
) -> T:
    """
    Execute an async zero-argument callable `fn` up to `attempts` times.

    Guarantees:
    - Retries only transient / retryable errors (as classified by is_retryable).
    - Stops immediately on non-retryable client errors (400, 401, etc.).
    - Raises the final exception if all attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

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

            logger.info(
                f"Attempt {attempt}/{attempts} failed with retryable error: {err}. Retrying..."
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected state: retry loop completed without returning or raising.")
