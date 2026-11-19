import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.db import RequestLog, get_db, init_db
from app.errors import ProviderError
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.routing.router import Router
from app.routing.strategy import PriorityStrategy
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist in the database
    init_db()

    # Configure connection pool limits and timeouts for upstream providers
    limits = httpx.Limits(
        max_keepalive_connections=20,  # keep up to 20 idle warm sockets
        max_connections=100,           # allow up to 100 concurrent requests
    )
    timeout = httpx.Timeout(
        timeout=60.0,  # overall operation deadline
        connect=5.0,   # max 5s to establish TCP/TLS connection
    )
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=timeout)

    yield

    # Shutdown: gracefully close connection pool
    if hasattr(app.state, "http_client") and app.state.http_client is not None:
        await app.state.http_client.aclose()


app = FastAPI(title="ModelMux", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Dependency: returns the shared AsyncClient from app.state
# ---------------------------------------------------------------------------
def get_http_client(request: Request) -> httpx.AsyncClient | None:
    """Return the shared connection-pooled AsyncClient from app.state."""
    return getattr(request.app.state, "http_client", None)


# ---------------------------------------------------------------------------
# Dependency: creates the router configured with the active strategy
# ---------------------------------------------------------------------------
def get_router(
    client: httpx.AsyncClient | None = Depends(get_http_client),
) -> Router:
    """Instantiate providers and return Router configured with PriorityStrategy."""
    providers: dict[str, BaseProvider] = {
        "groq": GroqProvider(client=client),
        "gemini": GeminiProvider(client=client),
        "ollama": OllamaProvider(client=client),
    }
    strategy = PriorityStrategy(priority=settings.priority_list)
    return Router(providers=providers, strategy=strategy)


# ---------------------------------------------------------------------------
# Dependency: returns the primary active provider from the Router.
# Tests can swap this via app.dependency_overrides[get_provider] = ...
# ---------------------------------------------------------------------------
def get_provider(
    router: Router = Depends(get_router),
) -> BaseProvider:
    """Return the top-priority provider determined by the router."""
    return router.get_primary_provider()


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
    Accept a ChatRequest, forward it to the active provider selected by the router,
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
