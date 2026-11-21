"""
Day 22 — Error Classification Unit Tests

Validates:
1. is_retryable(error): Decides if an error is transient (worth retrying) or permanent.
2. Parametrized table covering:
   - Retryable: Timeouts (504), Rate Limits (429), 5xx server errors, network connection drops.
   - Non-retryable: 4xx client errors (400 bad request, 401 unauthenticated, 403 forbidden, 404 not found).
   - Non-retryable: Malformed inputs, unsupported models, and general Python exceptions.
3. get_retry_after(error): Correctly extracts Retry-After seconds from ProviderRateLimited and HTTP responses.
"""

import httpx
import pytest

from app.errors import (
    ModelNotSupportedError,
    ProviderAuthenticationError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    get_retry_after,
    is_retryable,
)


def make_http_status_error(status_code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    """Helper to construct an httpx.HTTPStatusError with given status and headers."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=request,
        response=response,
    )


# ---------------------------------------------------------------------------
# Parametrized Test Table: Retryable vs. Non-Retryable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error,expected",
    [
        # --- RETRYABLE CASES (Transient failures) ---
        (ProviderTimeout("Timed out"), True),
        (TimeoutError("Async timeout"), True),
        (httpx.ReadTimeout("Socket read timed out"), True),
        (httpx.ConnectTimeout("TCP handshake timed out"), True),
        (httpx.NetworkError("Network unreachable"), True),
        (ProviderRateLimited("Rate limit hit"), True),
        (ProviderError("Internal error", status_code=500), True),
        (ProviderError("Bad Gateway", status_code=502), True),
        (ProviderError("Service Unavailable", status_code=503), True),
        (ProviderError("Gateway Timeout", status_code=504), True),
        (make_http_status_error(500), True),
        (make_http_status_error(502), True),
        (make_http_status_error(503), True),
        (make_http_status_error(429), True),

        # --- NON-RETRYABLE CASES (Permanent failures) ---
        (ModelNotSupportedError("Unsupported model"), False),
        (ProviderAuthenticationError("Invalid API key"), False),
        (ProviderBadRequest("Bad request arguments"), False),
        (ProviderError("Bad Request", status_code=400), False),
        (ProviderError("Unauthorized", status_code=401), False),
        (ProviderError("Forbidden", status_code=403), False),
        (ProviderError("Not Found", status_code=404), False),
        (make_http_status_error(400), False),
        (make_http_status_error(401), False),
        (make_http_status_error(403), False),
        (make_http_status_error(404), False),
        (ValueError("Invalid argument format"), False),
        (KeyError("missing_key"), False),
        (TypeError("int expected"), False),
    ],
)
def test_is_retryable_table(error, expected):
    """Ensure every error in the classification table matches expected retryability."""
    assert is_retryable(error) is expected


# ---------------------------------------------------------------------------
# Retry-After Header Extraction Tests
# ---------------------------------------------------------------------------

def test_get_retry_after_from_provider_rate_limited():
    """Extract retry_after seconds when attached to ProviderRateLimited."""
    err = ProviderRateLimited("Rate limited", provider="groq", retry_after=3.5)
    assert get_retry_after(err) == 3.5

    err_no_delay = ProviderRateLimited("Rate limited", provider="groq")
    assert get_retry_after(err_no_delay) is None


def test_get_retry_after_from_http_headers():
    """Extract Retry-After header from an HTTPStatusError response."""
    err = make_http_status_error(429, headers={"retry-after": "5"})
    assert get_retry_after(err) == 5.0

    err_no_header = make_http_status_error(429)
    assert get_retry_after(err_no_header) is None


def test_get_retry_after_returns_none_for_non_rate_limits():
    """Non-rate-limit errors return None for retry_after."""
    assert get_retry_after(ProviderTimeout("timeout")) is None
    assert get_retry_after(ValueError("bad value")) is None
