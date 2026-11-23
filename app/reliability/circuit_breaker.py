"""
Circuit Breaker Pattern for ModelMux (Days 28 & 29).

Protects the gateway and upstream providers by stopping calls to a provider
that is failing repeatedly, and automatically recovers via cool-down timers
and HALF-OPEN trial probes.

States:
- CLOSED: Normal operation. Requests flow freely to the provider.
- OPEN: Provider is failing. allow() returns False immediately to fail fast in 0ms.
- HALF_OPEN: Recovery trial state. Allows exactly ONE trial probe through to test health.
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

    # 7. Other unclassified exceptions default to True
    return True


class CircuitBreaker:
    """
    Circuit breaker state machine for an individual provider.

    Attributes:
        name: Identifier for the provider (e.g. "groq", "gemini").
        failure_threshold: Consecutive qualifying failures before tripping from CLOSED to OPEN.
        recovery_timeout: Duration in seconds to stay in OPEN before attempting HALF-OPEN recovery.
        trip_on_rate_limit: Whether HTTP 429 counts toward tripping the breaker.
        clock_fn: Time source function (injectable for deterministic testing).
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        trip_on_rate_limit: bool = False,
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be non-negative")

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.trip_on_rate_limit = trip_on_rate_limit
        self.clock_fn = clock_fn

        self._state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.opened_at: float | None = None
        self._half_open_probe_in_flight: bool = False

    def _check_and_update_state(self, now: float | None = None) -> CircuitState:
        """
        Dynamically transition OPEN -> HALF_OPEN once the cool-down timer expires.
        """
        if self._state == CircuitState.OPEN:
            current_time = now if now is not None else self.clock_fn()
            if self.opened_at is not None and (current_time - self.opened_at) >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_in_flight = False
                logger.info(
                    f"Circuit breaker for provider '{self.name}' cool-down ({self.recovery_timeout}s) "
                    f"elapsed: OPEN -> HALF_OPEN (ready for trial probe)."
                )
        return self._state

    @property
    def state(self) -> CircuitState:
        """Current state of the breaker, evaluating cool-down expiration automatically."""
        return self._check_and_update_state()

    @state.setter
    def state(self, value: CircuitState) -> None:
        self._state = value

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

        - CLOSED: Returns True (normal operation).
        - OPEN: Returns False (cool-down active, fast fail).
        - HALF_OPEN: Returns True for at most ONE trial probe concurrently.
                     Subsequent concurrent requests receive False until trial concludes.
        """
        st = self._check_and_update_state()

        if st == CircuitState.CLOSED:
            return True

        if st == CircuitState.OPEN:
            return False

        if st == CircuitState.HALF_OPEN:
            # Concurrency guard: guarantee at most one probe in flight
            if not self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = True
                logger.info(
                    f"Circuit breaker for provider '{self.name}': trial probe granted in HALF_OPEN state."
                )
                return True
            logger.debug(
                f"Circuit breaker for provider '{self.name}': trial probe already in flight, rejecting concurrent call."
            )
            return False

        return False

    def record_success(self) -> None:
        """
        Record a successful request.
        Resets consecutive failure count to 0.
        If in HALF_OPEN, closes the breaker and restores normal traffic.
        """
        prev_state = self._check_and_update_state()
        self.failure_count = 0
        self._state = CircuitState.CLOSED
        self.opened_at = None
        self._half_open_probe_in_flight = False

        if prev_state != CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker for provider '{self.name}' recovered: {prev_state.value} -> CLOSED."
            )

    def record_failure(self, error: Exception | None = None) -> bool:
        """
        Record a failed request.

        If error is provided, checks if it qualifies as an outage failure.
        Non-qualifying errors (like client 400s or 429s when excluded) do not increment counter.

        Returns:
            bool: True if this failure caused the breaker to transition/trip to OPEN, False otherwise.
        """
        if error is not None and not counts_toward_breaker(
            error, trip_on_rate_limit=self.trip_on_rate_limit
        ):
            # If a trial probe in HALF_OPEN failed on a client error (e.g. bad prompt),
            # release probe lock so future requests are not stuck
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_probe_in_flight = False
            logger.debug(
                f"Circuit breaker '{self.name}': ignoring non-qualifying error '{type(error).__name__}'."
            )
            return False

        now = self.clock_fn()
        self.last_failure_time = now
        st = self._check_and_update_state(now=now)

        if st == CircuitState.HALF_OPEN:
            # Trial probe failed! Immediately reopen breaker and reset cool-down timer
            self.trip(now=now)
            return True

        self.failure_count += 1
        if st == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.trip(now=now)
            return True

        return False

    def trip(self, now: float | None = None) -> None:
        """Manually or automatically trip the circuit breaker into the OPEN state."""
        self._state = CircuitState.OPEN
        self.opened_at = now if now is not None else self.clock_fn()
        self._half_open_probe_in_flight = False
        logger.warning(
            f"Circuit breaker for provider '{self.name}' TRIPPED to OPEN after "
            f"{self.failure_count} consecutive failure(s). Cool-down: {self.recovery_timeout}s."
        )

    def reset(self) -> None:
        """Reset the breaker to initial CLOSED state."""
        self._state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.opened_at = None
        self._half_open_probe_in_flight = False
        logger.info(f"Circuit breaker for provider '{self.name}' reset to CLOSED.")


# ---------------------------------------------------------------------------
# Global Circuit Breaker Registry
# ---------------------------------------------------------------------------
_CIRCUIT_BREAKERS: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    provider_name: str,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    trip_on_rate_limit: bool = False,
    clock_fn: Callable[[], float] = time.monotonic,
) -> CircuitBreaker:
    """
    Get or create a persistent CircuitBreaker instance for a given provider.
    """
    key = provider_name.lower().strip()
    if key not in _CIRCUIT_BREAKERS:
        _CIRCUIT_BREAKERS[key] = CircuitBreaker(
            name=key,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            trip_on_rate_limit=trip_on_rate_limit,
            clock_fn=clock_fn,
        )
    return _CIRCUIT_BREAKERS[key]


def reset_circuit_breakers() -> None:
    """Clear and reset all circuit breakers in the registry (useful for test isolation)."""
    _CIRCUIT_BREAKERS.clear()
