from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0, le=2)


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AttemptLog(BaseModel):
    attempt: int
    provider: str
    status: int | str
    duration_ms: float | int
    error: str | None = None
    error_category: str | None = None


class RequestLogResponse(BaseModel):
    id: int
    request_id: str | None = None
    tenant_id: str | None = None
    created_at: datetime
    provider: str
    model: str
    status: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    routing_decision: str | None = None
    attempts: list[dict[str, Any]] | None = None

    model_config = {"from_attributes": True}


# ============================================================================
# Admin API Schemas (Day 32)
# ============================================================================

class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    rate_limit: int = Field(default=60, ge=1)
    tenant_id: str | None = Field(default=None, max_length=50)


class CreateApiKeyResponse(BaseModel):
    id: int
    name: str
    tenant_id: str
    rate_limit: int
    key: str  # Plaintext key returned ONCE
    active: bool
    created_at: datetime


class RevokeApiKeyRequest(BaseModel):
    key: str | None = None
    key_id: int | None = None


class RevokeApiKeyResponse(BaseModel):
    status: str
    message: str
    key_id: int | None = None

