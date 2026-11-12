from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.db import init_db
from app.errors import ProviderError
from app.providers.base import BaseProvider
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    init_db()
    yield
    # Shutdown logic (if any)


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
) -> ChatResponse:
    """
    Accept a ChatRequest, forward it to the active provider,
    and return a ChatResponse.
    """
    return await provider.complete(request)