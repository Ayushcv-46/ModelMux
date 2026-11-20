"""
Defines routing strategies using the Strategy Pattern.
Supports:
1. PriorityStrategy: Orders providers by a fixed priority sequence.
2. RoundRobinStrategy: Cycles through providers evenly with each request.
3. ExplicitStrategy: Directs requests to the exact provider supporting the requested model.
"""

from abc import ABC, abstractmethod
from typing import Mapping

from app.errors import ModelNotSupportedError, ProviderError
from app.providers.base import BaseProvider
from app.schemas import ChatRequest


class RoutingStrategy(ABC):
    """Abstract Strategy interface for deciding provider order."""

    @abstractmethod
    def get_providers(
        self,
        providers: Mapping[str, BaseProvider],
        request: ChatRequest | None = None,
    ) -> list[BaseProvider]:
        """Return an ordered list of providers to use for this request."""
        pass


class PriorityStrategy(RoutingStrategy):
    """
    Orders providers based on a prioritized list of names.
    e.g., ["groq", "gemini", "ollama"]
    """

    def __init__(self, priority: list[str]):
        self.priority = [p.lower() for p in priority]

    def get_providers(
        self,
        providers: Mapping[str, BaseProvider],
        request: ChatRequest | None = None,
    ) -> list[BaseProvider]:
        """
        Return available providers matching the configured priority sequence.
        If a provider in the priority list doesn't exist in the registry, it is skipped.
        """
        ordered: list[BaseProvider] = []
        for name in self.priority:
            if name in providers:
                ordered.append(providers[name])

        # If priority list is empty or none matched, fallback to any remaining providers
        if not ordered:
            ordered = list(providers.values())

        return ordered


class RoundRobinStrategy(RoutingStrategy):
    """
    Cycles through available providers evenly, taking turns for each request.
    If providers are [groq, gemini, ollama]:
      - Call 1 -> [groq, gemini, ollama] (primary: groq)
      - Call 2 -> [gemini, ollama, groq] (primary: gemini)
      - Call 3 -> [ollama, groq, gemini] (primary: ollama)
      - Call 4 -> [groq, gemini, ollama] (repeats)
    """

    def __init__(self, provider_order: list[str] | None = None):
        self.provider_order = [p.lower() for p in provider_order] if provider_order else None
        self._index: int = 0

    def get_providers(
        self,
        providers: Mapping[str, BaseProvider],
        request: ChatRequest | None = None,
    ) -> list[BaseProvider]:
        if not providers:
            return []

        # Determine the order list of provider keys
        if self.provider_order:
            order = [p for p in self.provider_order if p in providers]
            if not order:
                order = list(providers.keys())
        else:
            order = list(providers.keys())

        # Rotate the sequence based on the current invocation count
        offset = self._index % len(order)
        self._index += 1

        rotated_keys = order[offset:] + order[:offset]
        return [providers[k] for k in rotated_keys]


class ExplicitStrategy(RoutingStrategy):
    """
    Routes directly to the provider that owns the model requested by the client.
    Guarantees:
      - Validates model/provider compatibility.
      - Never silently routes a requested model to an unrelated model or provider.
      - If no model is specified in the request, delegates to a fallback strategy.
    """

    def __init__(
        self,
        model_to_provider: Mapping[str, str],
        default_strategy: RoutingStrategy | None = None,
    ):
        self.model_to_provider = {k.lower(): v.lower() for k, v in model_to_provider.items()}
        self.default_strategy = default_strategy

    def get_providers(
        self,
        providers: Mapping[str, BaseProvider],
        request: ChatRequest | None = None,
    ) -> list[BaseProvider]:
        requested_model = request.model.strip() if request and request.model else None

        if requested_model:
            model_key = requested_model.lower()
            if model_key not in self.model_to_provider:
                raise ModelNotSupportedError(
                    message=f"Requested model '{requested_model}' is not supported by any registered provider.",
                    provider="router",
                )

            target_provider_name = self.model_to_provider[model_key]
            if target_provider_name not in providers:
                raise ProviderError(
                    message=f"Provider '{target_provider_name}' for model '{requested_model}' is unavailable.",
                    provider=target_provider_name,
                    status_code=503,
                )

            return [providers[target_provider_name]]

        # If no explicit model is specified, fall back to the default strategy
        if self.default_strategy:
            return self.default_strategy.get_providers(providers, request)

        return list(providers.values())
