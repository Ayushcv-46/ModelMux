"""
Custom exceptions and error classification for ModelMux.
Provides:
- ProviderError hierarchy (ProviderTimeout, ProviderRateLimited, etc.)
- is_retryable(error): Single source of truth for transient vs. permanent error classification.
- get_retry_after(error): Extracts upstream Retry-After delay if provided.
"""

from typing import Any
import httpx


class ProviderError(Exception):
    """Something went wrong at the provider (parent of all provider errors)."""

    status_code: int = 502

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        if status_code is not None:
            self.status_code = status_code


class ProviderTimeout(ProviderError):
    """Upstream provider took too long to answer."""

    status_code: int = 504


class ProviderRateLimited(ProviderError):
    """Upstream provider rejected call with HTTP 429 Too Many Requests."""

    status_code: int = 429

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retry_after: float | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message, provider=provider, status_code=status_code)
        self.retry_after = retry_after


class ModelNotSupportedError(ProviderError):
    """Client requested a model not supported by any registered provider."""

    status_code: int = 400


class ProviderAuthenticationError(ProviderError):
    """Invalid credentials or missing permissions at upstream provider."""

    status_code: int = 401


class ProviderBadRequest(ProviderError):
    """Malformed request or invalid arguments sent to provider."""

    status_code: int = 400


class AllProvidersFailed(ProviderError):
    """All candidate providers in the fallback chain failed to satisfy the request."""

    status_code: int = 503

    def __init__(
        self,
        message: str = "All candidate providers failed to fulfill the request",
        provider: str = "router",
        details: list[dict] | None = None,
    ):
        super().__init__(message, provider=provider, status_code=503)
        self.details = details or []


def is_retryable(error: Exception) -> bool:
    """
    Decides whether an error is transient (worth retrying) or permanent (fail immediately).

    Retryable (True):
    - Timeouts (HTTP 504, TimeoutError, httpx.TimeoutException)
    - Rate limits (HTTP 429)
    - Upstream server errors (HTTP 5xx: 500, 502, 503)
    - Network drops / connection errors (httpx.NetworkError)

    Non-retryable (False):
    - Client / input errors (HTTP 4xx: 400, 401, 403, 404, etc.)
    - Unsupported models / malformed payloads
    - General coding errors (ValueError, KeyError, TypeError, etc.)
    """
    # 1. Any Timeout error
    if isinstance(error, (ProviderTimeout, TimeoutError, httpx.TimeoutException)):
        return True

    # 2. Rate limited (429)
    if isinstance(error, ProviderRateLimited):
        return True

    # 3. ProviderError subclasses or instances
    if isinstance(error, ProviderError):
        code = getattr(error, "status_code", 502)
        # 429 is retryable
        if code == 429:
            return True
        # 5xx (500 to 599) are retryable
        if 500 <= code < 600:
            return True
        # 4xx client errors are permanent
        return False

    # 4. HTTPX network and status errors
    if isinstance(error, (httpx.NetworkError, httpx.ConnectError)):
        return True

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 429 or 500 <= status < 600:
            return True
        return False

    # All other general exceptions are not retryable
    return False


def get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header or parameter in seconds, if available."""
    if isinstance(error, ProviderRateLimited) and error.retry_after is not None:
        return error.retry_after

    if isinstance(error, httpx.HTTPStatusError):
        header_val = error.response.headers.get("retry-after")
        if header_val:
            try:
                return float(header_val)
            except ValueError:
                return None

    return None
