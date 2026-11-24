from typing import Annotated
import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
import httpx
import redis.asyncio as aioredis
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.auth import (
    create_api_key,
    get_current_api_key,
    get_optional_api_key,
    hash_api_key,
    verify_admin_secret,
)
from app.config import load_yaml_config, settings
from app.db import ApiKey, RequestLog, get_db, init_db
from app.errors import ProviderError
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.reliability import execute_fallback_chain, execute_with_timeout, retry
from app.routing.router import Router
from app.routing.strategy import (
    ExplicitStrategy,
    PriorityStrategy,
    RoundRobinStrategy,
    RoutingStrategy,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    RequestLogResponse,
    RevokeApiKeyRequest,
    RevokeApiKeyResponse,
)


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

    # Initialize Async Redis connection (Day 33)
    cfg = load_yaml_config()
    redis_url = getattr(settings, "redis_url", None) or cfg.redis.url
    try:
        redis_client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=cfg.redis.connect_timeout,
        )
        await redis_client.ping()
        app.state.redis = redis_client
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as exc:
        app.state.redis = None
        logger.warning(
            f"Could not connect to Redis at {redis_url}: {exc}. Running in degraded mode without Redis."
        )

    yield

    # Shutdown: gracefully close Redis and HTTP client
    if hasattr(app.state, "redis") and app.state.redis is not None:
        try:
            await app.state.redis.aclose()
        except Exception:
            pass

    if hasattr(app.state, "http_client") and app.state.http_client is not None:
        await app.state.http_client.aclose()


app = FastAPI(title="ModelMux", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Dependency: returns the shared AsyncClient from app.state
# ---------------------------------------------------------------------------
def get_http_client(request: Request) -> httpx.AsyncClient | None:
    """Return the shared connection-pooled AsyncClient from app.state."""
    return getattr(request.app.state, "http_client", None)


def get_redis(request: Request) -> aioredis.Redis | None:
    """Return the shared Async Redis client from app.state, or None if unavailable."""
    return getattr(request.app.state, "redis", None)


# ---------------------------------------------------------------------------
# Dependency: creates the router configured with the active strategy
# ---------------------------------------------------------------------------
def get_router(
    client: httpx.AsyncClient | None = Depends(get_http_client),
) -> Router:
    """
    Instantiate providers and return Router configured with the strategy
    specified in config.yaml (PriorityStrategy, RoundRobinStrategy, or ExplicitStrategy).
    """
    providers: dict[str, BaseProvider] = {
        "groq": GroqProvider(client=client),
        "gemini": GeminiProvider(client=client),
        "ollama": OllamaProvider(client=client),
        "mock": MockProvider(),
    }

    cfg = load_yaml_config()
    strategy_name = cfg.routing.strategy.lower()

    if strategy_name == "round_robin":
        strategy = RoundRobinStrategy(provider_order=cfg.routing.priority)
    elif strategy_name == "explicit":
        fallback_strategy = PriorityStrategy(priority=cfg.routing.priority)
        strategy = ExplicitStrategy(
            model_to_provider=cfg.model_to_provider,
            default_strategy=fallback_strategy,
        )
    else:
        # Default: PriorityStrategy
        strategy = PriorityStrategy(priority=cfg.routing.priority)

    return Router(providers=providers, strategy=strategy)


# ---------------------------------------------------------------------------
# Dependency: returns the primary active provider from the Router.
# Tests can swap this via app.dependency_overrides[get_provider] = ...
# ---------------------------------------------------------------------------
def get_provider(
    request: ChatRequest | None = None,
    client: httpx.AsyncClient | None = Depends(get_http_client),
    router: Router | None = Depends(get_router),
) -> BaseProvider:
    """Return the top-priority provider determined by the router and request."""
    if not isinstance(router, Router):
        router = get_router(client=client)
    return router.get_primary_provider(request=request)


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
async def health(request: Request) -> dict[str, str]:
    """Health check endpoint reporting application and Redis connection status."""
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is not None:
        try:
            pong = await redis_client.ping()
            if pong:
                return {"status": "ok", "redis": "up"}
        except Exception:
            return {"status": "degraded", "redis": "down"}
    return {"status": "degraded", "redis": "down"}


# ---------------------------------------------------------------------------
# Admin Routes (Day 32)
# ---------------------------------------------------------------------------
@app.post("/admin/keys", response_model=CreateApiKeyResponse)
def admin_create_key(
    body: CreateApiKeyRequest,
    admin: str = Depends(verify_admin_secret),
    db: Session = Depends(get_db),
) -> CreateApiKeyResponse:
    """Create a new client API key. Protected by ADMIN_SECRET."""
    record, plain_key = create_api_key(
        db=db,
        name=body.name,
        rate_limit=body.rate_limit,
        tenant_id=body.tenant_id,
    )
    return CreateApiKeyResponse(
        id=record.id,
        name=record.name,
        tenant_id=record.tenant_id or record.name,
        rate_limit=record.rate_limit,
        key=plain_key,  # Returned ONCE
        active=record.active,
        created_at=record.created_at,
    )


@app.post("/admin/keys/revoke", response_model=RevokeApiKeyResponse)
def admin_revoke_key(
    body: RevokeApiKeyRequest,
    admin: str = Depends(verify_admin_secret),
    db: Session = Depends(get_db),
) -> RevokeApiKeyResponse:
    """Revoke an active API key by key_id or key string. Protected by ADMIN_SECRET."""
    query = db.query(ApiKey)
    if body.key_id is not None:
        record = query.filter(ApiKey.id == body.key_id).first()
    elif body.key is not None and body.key.strip():
        kh = hash_api_key(body.key)
        record = query.filter(ApiKey.key_hash == kh).first()
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'key_id' or 'key' to revoke.")

    if record is None:
        raise HTTPException(status_code=404, detail="API key not found.")

    record.active = False
    db.commit()

    return RevokeApiKeyResponse(
        status="revoked",
        message=f"API key '{record.name}' (id={record.id}) has been revoked.",
        key_id=record.id,
    )


# ---------------------------------------------------------------------------
# Core Chat Route (Day 8 / 32)
# ---------------------------------------------------------------------------
@app.post("/v1/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    req: Request = None,
    api_key: ApiKey = Depends(get_current_api_key),
    provider: BaseProvider = Depends(get_provider),
    router: Router = Depends(get_router),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Accept a ChatRequest, forward it to the active provider selected by the router,
    log the outcome (success or failure) to the database,
    and return the ChatResponse.
    Protected by API Key authentication dependency.
    """
    start_time = time.perf_counter()
    model_name = request.model or getattr(provider, "name", "unknown")
    request_id = (
        req.headers.get("x-request-id")
        if req and hasattr(req, "headers") and req.headers.get("x-request-id")
        else f"req_{uuid.uuid4().hex[:12]}"
    )
    routing_decision = (
        getattr(router.strategy, "name", router.strategy.__class__.__name__)
        if router and hasattr(router, "strategy")
        else "Direct"
    )

    cfg = load_yaml_config()
    if get_provider in app.dependency_overrides:
        candidate_providers = [provider]
    elif router:
        candidate_providers = router.get_providers(request=request)
    else:
        candidate_providers = [provider]

    tenant_id = api_key.tenant_id if isinstance(api_key, ApiKey) else "default"
    attempts_log: list[dict] = []
    try:
        response = await execute_fallback_chain(
            providers=candidate_providers,
            request=request,
            cfg=cfg,
            attempts_log=attempts_log,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Log successful completion
        log_entry = RequestLog(
            request_id=request_id,
            tenant_id=tenant_id,
            provider=response.provider,
            model=response.model,
            status="success",
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            routing_decision=routing_decision,
            attempts=attempts_log,
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
            request_id=request_id,
            tenant_id=tenant_id,
            provider=exc.provider or getattr(provider, "name", "unknown"),
            model=model_name,
            status=type(exc).__name__,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=0,
            routing_decision=routing_decision,
            attempts=attempts_log,
        )
        db.add(log_entry)
        db.commit()

        # Re-raise so provider_error_handler returns the proper HTTP error response
        raise exc


@app.get("/v1/requests", response_model=list[RequestLogResponse])
@app.get("/requests", response_model=list[RequestLogResponse])
def get_requests(
    provider: Annotated[str | None, Query(description="Filter by provider name (e.g. 'groq')")] = None,
    status: Annotated[str | None, Query(description="Filter by status (e.g. 'success', 'ProviderTimeout')")] = None,
    tenant_id: Annotated[str | None, Query(description="Filter by tenant ID (admin only)")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Max logs to return (1-100)")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of logs to skip")] = 0,
    db: Session = Depends(get_db),
    api_key: ApiKey | None = Depends(get_optional_api_key),
) -> list[RequestLog]:
    """
    Retrieve request history with filtering by provider/status/tenant and pagination (limit/offset).
    Tenant Scoping: Non-admin tenant keys are strictly scoped to their own tenant_id.
    """
    query = db.query(RequestLog)

    # Tenant scoping (Engineering refinement)
    if isinstance(api_key, ApiKey):
        scoped_tenant = api_key.tenant_id or api_key.name
        query = query.filter(RequestLog.tenant_id == scoped_tenant)
    elif tenant_id is not None and isinstance(tenant_id, str) and tenant_id.strip():
        query = query.filter(RequestLog.tenant_id == tenant_id.strip())

    if provider is not None and isinstance(provider, str) and provider.strip():
        query = query.filter(RequestLog.provider.ilike(provider.strip()))
    if status is not None and isinstance(status, str) and status.strip():
        query = query.filter(RequestLog.status.ilike(status.strip()))

    return query.order_by(RequestLog.id.desc()).offset(offset).limit(limit).all()
