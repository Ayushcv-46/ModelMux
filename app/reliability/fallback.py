"""
Fallback chain for ModelMux.
Implements Graceful Degradation:
Iterates over candidate providers in router priority order.
Returns the first successful response.
If all fail, raises AllProvidersFailed (HTTP 503).
"""

import logging
import time
from typing import Any, Awaitable, Callable

from app.config import AppConfig, load_yaml_config
from app.errors import AllProvidersFailed
from app.providers.base import BaseProvider
from app.reliability.retry import retry
from app.reliability.sanitizer import sanitize_error
from app.reliability.timeout import execute_with_timeout
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


async def execute_fallback_chain(
    providers: list[BaseProvider],
    request: ChatRequest,
    cfg: AppConfig | None = None,
    sleep_fn: Callable[[float], Awaitable[Any]] | None = None,
    clock_fn: Callable[[], float] = time.perf_counter,
    attempts_log: list[dict] | None = None,
) -> ChatResponse:
    """
    Execute requests across candidate providers in priority order.

    Guarantees:
    - Returns the first successful response.
    - If a provider fails after retries, logs the failure and falls back to the next provider.
    - Skips providers incompatible with request.model.
    - All providers and retries share the single total_timeout_seconds budget.
    - If all candidate providers fail, raises AllProvidersFailed (HTTP 503).
    - Populates attempts_log with granular execution history (order, provider, status, duration, error).
    """
    if not providers:
        raise AllProvidersFailed("No candidate providers available for fallback.")

    config = cfg or load_yaml_config()
    rel_cfg = config.reliability
    model_to_prov = config.model_to_provider
    total_budget = rel_cfg.total_timeout_seconds
    start_time = clock_fn()

    recorded_attempts = attempts_log if attempts_log is not None else []
    errors: list[dict] = []
    last_exception: Exception | None = None

    for provider in providers:
        provider_name = getattr(provider, "name", "unknown")

        # 1. Enforce single deadline: check if overall time budget is spent
        elapsed = clock_fn() - start_time
        remaining_budget = total_budget - elapsed
        if remaining_budget <= 0:
            logger.warning(
                f"Total request budget of {total_budget:.2f}s exhausted before trying provider '{provider_name}'."
            )
            break

        # 2. Model compatibility guard:
        # If client requested a specific model, verify this provider supports it
        if request.model and model_to_prov:
            req_model = request.model.strip().lower()
            expected_owner = model_to_prov.get(req_model)
            if expected_owner and expected_owner != provider_name.lower():
                logger.info(
                    f"Skipping provider '{provider_name}': incompatible with requested model '{request.model}'."
                )
                continue

        # 3. Determine attempt timeout: bounded by remaining budget
        prov_timeout = config.get_provider_timeout(provider_name, default=30.0)
        effective_timeout = min(prov_timeout, remaining_budget)

        logger.info(f"Attempting provider '{provider_name}' (effective timeout: {effective_timeout:.2f}s)...")

        def make_record_attempt(prov_name: str):
            def record_attempt(status: Any, duration_ms: int, error: Exception | None):
                category = type(error).__name__ if error is not None else None
                recorded_attempts.append({
                    "attempt": len(recorded_attempts) + 1,
                    "provider": prov_name,
                    "status": status,
                    "duration_ms": duration_ms,
                    "error": sanitize_error(error) if error is not None else None,
                    "error_category": category,
                })
            return record_attempt

        try:
            # Execute with per-attempt timeout and retries, bounded by remaining budget
            response = await retry(
                lambda: execute_with_timeout(provider, request, effective_timeout),
                attempts=rel_cfg.max_attempts,
                base_delay=rel_cfg.base_delay,
                max_delay=rel_cfg.max_delay,
                total_timeout_seconds=remaining_budget,
                sleep_fn=sleep_fn,
                clock_fn=clock_fn,
                on_attempt=make_record_attempt(provider_name),
            )
            # Return first success immediately
            logger.info(f"Provider '{provider_name}' succeeded. Returning response.")
            return response

        except Exception as err:
            logger.warning(
                f"Provider '{provider_name}' failed after attempts: {err}. Falling back to next candidate..."
            )
            errors.append({"provider": provider_name, "error": str(err)})
            last_exception = err

    # If only a single provider was attempted and failed, preserve its specific error (504, 429, etc.)
    if len(providers) == 1 and last_exception is not None:
        raise last_exception

    # If multiple providers were tried and all failed, return 503
    error_summary = "; ".join(f"[{e['provider']}] {e['error']}" for e in errors) if errors else "Budget exhausted"
    raise AllProvidersFailed(
        message=f"All candidate providers failed to fulfill the request. ({error_summary})",
        provider="router",
        details=errors,
    )
