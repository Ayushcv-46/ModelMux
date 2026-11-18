import time
from typing import Any
import httpx

from app.config import settings
from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.base import BaseProvider
from app.schemas import ChatRequest, ChatResponse


class GeminiProvider(BaseProvider):
    """Adapter connecting ModelMux to Google Gemini API."""

    name: str = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-3.5-flash-lite",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.default_model = default_model
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected client or create a new one."""
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self.timeout)

    def _build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], str]:
        """
        Translate ModelMux ChatRequest into Gemini REST API payload.
        Maps 'assistant' role to 'model', and 'system' role to 'systemInstruction'.
        """
        model_name = request.model or self.default_model
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for msg in request.messages:
            if msg.role == "system":
                # Google Gemini accepts system instructions in a dedicated field
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "assistant":
                # Gemini calls assistant responses 'model'
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})

        # Ensure contents is never empty (if only system prompt was provided)
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        return payload, model_name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send ChatRequest to Gemini generateContent and return normalized ChatResponse."""
        if not self.api_key:
            raise ProviderError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in .env file.",
                provider=self.name,
            )

        payload, model_name = self._build_payload(request)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        params = {"key": self.api_key}
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            if self._client is not None:
                response = await client.post(url, params=params, json=payload)
            else:
                async with client:
                    response = await client.post(url, params=params, json=payload)

        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderTimeout(
                f"Gemini request timed out after {self.timeout}s: {exc}",
                provider=self.name,
            ) from exc

        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Failed to connect to Gemini API: {exc}",
                provider=self.name,
            ) from exc

        except httpx.RequestError as exc:
            raise ProviderError(
                f"Network error communicating with Gemini API: {exc}",
                provider=self.name,
            ) from exc

        # Map Google HTTP error status codes to ModelMux exceptions
        if response.status_code == 429:
            raise ProviderRateLimited(
                f"Gemini rate limit exceeded: {response.text}",
                provider=self.name,
            )

        if response.status_code >= 500:
            raise ProviderError(
                f"Gemini service unavailable (HTTP {response.status_code}): {response.text}",
                provider=self.name,
            )

        if response.status_code != 200:
            raise ProviderError(
                f"Gemini API returned error (HTTP {response.status_code}): {response.text}",
                provider=self.name,
            )

        data = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract generated text from candidates
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        # Extract token counts from usageMetadata
        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0) or 0
        tokens_out = usage.get("candidatesTokenCount", 0) or 0

        return ChatResponse(
            text=text,
            provider=self.name,
            model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
