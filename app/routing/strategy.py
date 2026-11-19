"""
Defines routing strategies using the Strategy Pattern.
"""

from abc import ABC, abstractmethod
from typing import Mapping

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
