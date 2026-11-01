from abc import ABC, abstractmethod

from app.schemas import ChatRequest, ChatResponse


class BaseProvider(ABC):
    """Contract every provider (Mock, Ollama, Gemini, Groq) must follow."""

    name: str

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Take a ChatRequest, return a ChatResponse, or raise a ProviderError."""
        ...