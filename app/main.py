from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.errors import ProviderError
from app.providers.base import BaseProvider
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="ModelMux")


# ---------------------------------------------------------------------------
# Dependency: returns the active provider.
# Tests can swap this via app.dependency_overrides[get_provider] = ...
# ---------------------------------------------------------------------------
def get_provider() -> BaseProvider:
    """Return the default provider for every request."""
    return MockProvider()


# ---------------------------------------------------------------------------
# Exception handler: turns any ProviderError into a structured JSON response.
# (Already existed — untouched.)
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

    Any ProviderError raised inside provider.complete() is caught
    automatically by the exception handler above.
    """
    return await provider.complete(request)