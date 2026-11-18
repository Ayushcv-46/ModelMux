import time
from typing import Any
import httpx

from app.config import settings
from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


class GroqProvider(BaseProvider):
    """Adapter connecting ModelMux to Groq via its OpenAI-compatible REST API."""

    name: str = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "openai/gpt-oss-20b",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.default_model = default_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected client or create a new one."""
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self.timeout)

    def _build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], str]:
        """Translate ModelMux ChatRequest into standard OpenAI-compatible payload."""
        model_name = request.model or self.default_model
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        return payload, model_name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send request to Groq chat completions endpoint and return ChatResponse."""
        if not self.api_key:
            raise ProviderError(
                "Groq API key is not configured. Set GROQ_API_KEY in .env file.",
                provider=self.name,
            )

        payload, model_name = self._build_payload(request)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            if self._client is not None:
                response = await client.post(url, headers=headers, json=payload)
            else:
                async with client:
                    response = await client.post(url, headers=headers, json=payload)

        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderTimeout(
                f"Groq request timed out after {self.timeout}s: {exc}",
                provider=self.name,
            ) from exc

        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Failed to connect to Groq API: {exc}",
                provider=self.name,
            ) from exc

        except httpx.RequestError as exc:
            raise ProviderError(
                f"Network error communicating with Groq API: {exc}",
                provider=self.name,
            ) from exc

        # Map error status codes to ModelMux exceptions
        if response.status_code == 429:
            raise ProviderRateLimited(
                f"Groq rate limit exceeded: {response.text}",
                provider=self.name,
            )

        if response.status_code >= 500:
            raise ProviderError(
                f"Groq service unavailable (HTTP {response.status_code}): {response.text}",
                provider=self.name,
            )

        if response.status_code != 200:
            raise ProviderError(
                f"Groq API returned error (HTTP {response.status_code}): {response.text}",
                provider=self.name,
            )

        data = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract generated content from choices
        choices = data.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or msg.get("reasoning", "") or msg.get("reasoning_content", "")

        # Extract token usage
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0) or 0
        tokens_out = usage.get("completion_tokens", 0) or 0

        return ChatResponse(
            text=text,
            provider=self.name,
            model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
