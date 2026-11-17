import os
import time
from typing import Any
import httpx

from app.errors import ProviderError, ProviderTimeout
from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


class OllamaProvider(BaseProvider):
    """Adapter connecting ModelMux to a local Ollama instance via HTTP."""

    name: str = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str = "qwen3:4b",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected client or a new one."""
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self.timeout)

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Translate ModelMux ChatRequest into Ollama's /api/chat payload."""
        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        return payload

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send ChatRequest to Ollama /api/chat and return normalized ChatResponse."""
        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(request)
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            # If client was injected externally, we do not close it; otherwise we close it.
            if self._client is not None:
                response = await client.post(url, json=payload)
            else:
                async with client:
                    response = await client.post(url, json=payload)

        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderTimeout(
                f"Ollama request timed out after {self.timeout}s: {exc}",
                provider=self.name,
            ) from exc

        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Failed to connect to Ollama at {self.base_url}. Is Ollama running? Error: {exc}",
                provider=self.name,
            ) from exc

        except httpx.RequestError as exc:
            raise ProviderError(
                f"Network error communicating with Ollama: {exc}",
                provider=self.name,
            ) from exc

        # Handle non-200 HTTP responses from Ollama server
        if response.status_code != 200:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {response.text}",
                provider=self.name,
            )

        data = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract message text and token statistics
        message = data.get("message", {})
        text = message.get("content", "") or message.get("thinking", "")
        tokens_in = data.get("prompt_eval_count", 0) or 0
        tokens_out = data.get("eval_count", 0) or 0
        model_used = data.get("model", request.model or self.default_model)

        return ChatResponse(
            text=text,
            provider=self.name,
            model=model_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
