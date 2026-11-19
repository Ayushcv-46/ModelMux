"""
Router module: coordinates strategies and provider instances.
"""

from typing import Mapping

from app.providers.base import BaseProvider
from app.routing.strategy import PriorityStrategy, RoutingStrategy
from app.schemas import ChatRequest


class Router:
    """
    Context class in the Strategy Pattern.
    Delegates provider ranking to its configured RoutingStrategy.
    """

    def __init__(
        self,
        providers: Mapping[str, BaseProvider],
        strategy: RoutingStrategy | None = None,
    ):
        self.providers = providers
        # Default strategy: priority matching all registered providers
        self.strategy = strategy or PriorityStrategy(priority=list(providers.keys()))

    def get_providers(self, request: ChatRequest | None = None) -> list[BaseProvider]:
        """Return the ordered list of providers selected by the active strategy."""
        return self.strategy.get_providers(self.providers, request)

    def get_primary_provider(self, request: ChatRequest | None = None) -> BaseProvider:
        """Convenience method returning the 1st provider in the strategy's order."""
        providers = self.get_providers(request)
        if not providers:
            raise RuntimeError("No available providers configured in Router.")
        return providers[0]
