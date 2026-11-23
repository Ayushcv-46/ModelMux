"""
Circuit Breaker Pattern for ModelMux (Day 28).

Protects the gateway and upstream providers by stopping calls to a provider
that is failing repeatedly.

States:
- CLOSED: Normal operation. Requests flow freely to the provider.
- OPEN: Provider is failing. allow() returns False immediately to fail fast without touching the network.
- HALF_OPEN: Recovery trial state (handled in Day 29).
"""

from enum import Enum
import logging
import time
from typing import Callable

import httpx

from app.errors import (
    ProviderAuthenticationError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """The three canonical states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def counts_toward_breaker(error: Exception, trip_on_rate_limit: bool = False) -> bool:
    """
    Determine whether a given exception represents an upstream outage/failure
    that should count toward opening the circuit breaker.

    Counts toward breaker (True):
    - Timeouts (ProviderTimeout, TimeoutError, httpx.TimeoutException)
    - 5xx Server Errors (ProviderError 5xx, httpx.HTTPStatusError 5xx)
    - Network / Connection drops (httpx.NetworkError, ConnectionError)

    Does NOT count toward breaker (False):
    - 4xx Client Errors (ProviderBadRequest, ProviderAuthenticationError, 400-499)
      Reason: A bad prompt or malformed payload from a user is not a provider outage.
    - 429 Rate Limits (by default, unless trip_on_rate_limit=True)
      Reason: Rate limits mean the provider is working but throttling; backoff/retry handles this.
    """
    # 1. Timeouts indicate provider impairment
    if isinstance(error, (ProviderTimeout, TimeoutError, httpx.TimeoutException)):
        return True

    # 2. Rate limiting (429) - separate treatment
    if isinstance(error, ProviderRateLimited):
        return trip_on_rate_limit

    # 3. Known client errors (400, 401, etc.)
    if isinstance(error, (ProviderBadRequest, ProviderAuthenticationError)):
        return False

    # 4. ProviderError hierarchy
    if isinstance(error, ProviderError):
        code = getattr(error, "status_code", 502)
        if code == 429:
            return trip_on_rate_limit
        if 400 <= code < 500:
            return False
        if 500 <= code < 600:
            return True
        return True

    # 5. Network / Transport errors
    if isinstance(error, (httpx.NetworkError, httpx.ConnectError, ConnectionError)):
        return True

    # 6. HTTPX HTTP status errors
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 429:
            return trip_on_rate_limit
        if 400 <= status < 500:
            return False
        if 500 <= status < 600:
            return True
        return False

    # 7. Other unclassified exceptions default to True (e.g. unexpected runtime crash during provider call)
    return True


class CircuitBreaker:
    """
    Circuit breaker state machine for an individual provider.

    Attributes:
        name: Identifier for the provider (e.g. "groq", "gemini").
        failure_threshold: Consecutive qualifying failures before tripping from CLOSED to OPEN.
        trip_on_rate_limit: Whether HTTP 429 counts toward tripping the breaker.
        clock_fn: Time source function (injectable for deterministic testing).
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        trip_on_rate_limit: bool = False,
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")

        self.name = name
        self.failure_threshold = failure_threshold
        self.trip_on_rate_limit = trip_on_rate_limit
        self.clock_fn = clock_fn

        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.opened_at: float | None = None

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def allow(self) -> bool:
        """
        Check if requests are allowed to proceed to this provider.

        - CLOSED: True (normal traffic)
        - OPEN: False (fast fail, circuit broken)
        - HALF_OPEN: True (cautious probe)
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            return False
        # HALF_OPEN allows a trial probe (extended in Day 29)
        return True

    def record_success(self) -> None:
        """
        Record a successful request.
        Resets consecutive failure count to 0.
        If in HALF_OPEN, closes the breaker.
        """
        prev_state = self.state
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

        if prev_state != CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker for provider '{self.name}' recovered: {prev_state.value} -> CLOSED."
            )

    def record_failure(self, error: Exception | None = None) -> bool:
        """
        Record a failed request.

        If error is provided, checks if it qualifies as an outage failure.
        Non-qualifying errors (like client 400s or 429s when excluded) do not increment the counter.

        Returns:
            bool: True if this failure caused the breaker to transition/trip to OPEN, False otherwise.
        """
        if error is not None and not counts_toward_breaker(
            error, trip_on_rate_limit=self.trip_on_rate_limit
        ):
            logger.debug(
                f"Circuit breaker '{self.name}': ignoring non-qualifying error '{type(error).__name__}'."
            )
            return False

        self.failure_count += 1
        now = self.clock_fn()
        self.last_failure_time = now

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.trip(now=now)
                return True
        elif self.state == CircuitState.HALF_OPEN:
            # A trial failure immediately re-trips the breaker
            self.trip(now=now)
            return True

        return False

    def trip(self, now: float | None = None) -> None:
        """Manually or automatically trip the circuit breaker into the OPEN state."""
        self.state = CircuitState.OPEN
        self.opened_at = now if now is not None else self.clock_fn()
        logger.warning(
            f"Circuit breaker for provider '{self.name}' TRIPPED to OPEN after "
            f"{self.failure_count} consecutive failure(s). Further calls blocked."
        )

    def reset(self) -> None:
        """Reset the breaker to initial CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.opened_at = None
        logger.info(f"Circuit breaker for provider '{self.name}' reset to CLOSED.")
