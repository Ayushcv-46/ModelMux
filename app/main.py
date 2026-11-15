import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import RequestLog, get_db, init_db
from app.errors import ProviderError
from app.providers.base import BaseProvider
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist in the database
    init_db()
    yield
    # Shutdown logic (if any needed in future)


app = FastAPI(title="ModelMux", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Dependency: returns the active provider.
# Tests can swap this via app.dependency_overrides[get_provider] = ...
# ---------------------------------------------------------------------------
def get_provider() -> BaseProvider:
    """Return the default provider for every request."""
    return MockProvider()


# ---------------------------------------------------------------------------
# Exception handler: turns any ProviderError into a structured JSON response.
# ---------------------------------------------------------------------------
@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": type(exc).__name__,
                "message": exc.message,
                "provider": exc.provider,
            }
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: BaseProvider = Depends(get_provider),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Accept a ChatRequest, forward it to the active provider,
    log the outcome (success or failure) to the database,
    and return the ChatResponse.
    """
    start_time = time.perf_counter()
    model_name = request.model or getattr(provider, "name", "unknown")

    try:
        response = await provider.complete(request)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Log successful completion
        log_entry = RequestLog(
            provider=response.provider,
            model=response.model,
            status="success",
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )
        db.add(log_entry)
        db.commit()

        return response

    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Estimate input tokens from prompt even on failure
        tokens_in = sum(len(m.content.split()) for m in request.messages)

        # Log failure with error type (ensure provider is never None)
        log_entry = RequestLog(
            provider=exc.provider or getattr(provider, "name", "unknown"),
            model=model_name,
            status=type(exc).__name__,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=0,
        )
        db.add(log_entry)
        db.commit()

        # Re-raise so provider_error_handler returns the proper HTTP error response
        raise exc