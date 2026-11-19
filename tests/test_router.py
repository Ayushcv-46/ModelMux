"""
Day 18 — Router & PriorityStrategy Unit Tests

Validates:
1. PriorityStrategy respects configured ordering.
2. Router delegates ranking to strategy and returns primary provider.
3. Changing priority changes which provider is returned first.
4. Gracefully skips missing or unregistered providers.
"""

import pytest

from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.routing.router import Router
from app.routing.strategy import PriorityStrategy
from app.schemas import ChatRequest, Message


@pytest.fixture
def sample_providers():
    return {
        "groq": GroqProvider(api_key="fake-groq-key"),
        "gemini": GeminiProvider(api_key="fake-gemini-key"),
        "ollama": OllamaProvider(),
    }


def test_priority_strategy_respects_custom_order(sample_providers):
    """PriorityStrategy should order providers exactly matching the input priority list."""
    # Priority: gemini first, then ollama, then groq
    strategy = PriorityStrategy(priority=["gemini", "ollama", "groq"])
    router = Router(providers=sample_providers, strategy=strategy)

    ordered = router.get_providers()
    names = [p.name for p in ordered]

    assert names == ["gemini", "ollama", "groq"]
    assert router.get_primary_provider().name == "gemini"


def test_changing_priority_changes_primary_provider(sample_providers):
    """Switching priority list from groq-first to ollama-first changes the primary provider."""
    # 1. Groq first
    router1 = Router(
        providers=sample_providers,
        strategy=PriorityStrategy(priority=["groq", "gemini", "ollama"]),
    )
    assert router1.get_primary_provider().name == "groq"

    # 2. Change priority to ollama first
    router2 = Router(
        providers=sample_providers,
        strategy=PriorityStrategy(priority=["ollama", "groq", "gemini"]),
    )
    assert router2.get_primary_provider().name == "ollama"


def test_priority_strategy_skips_unknown_providers(sample_providers):
    """If priority list mentions an unregistered provider, it is safely skipped."""
    strategy = PriorityStrategy(priority=["non_existent_llm", "groq", "gemini"])
    router = Router(providers=sample_providers, strategy=strategy)

    ordered = router.get_providers()
    names = [p.name for p in ordered]

    assert names == ["groq", "gemini"]
    assert router.get_primary_provider().name == "groq"


def test_router_empty_providers_raises_runtime_error():
    """An empty router raises RuntimeError when primary provider is requested."""
    router = Router(providers={}, strategy=PriorityStrategy(priority=[]))
    with pytest.raises(RuntimeError) as exc_info:
        router.get_primary_provider()

    assert "No available providers" in str(exc_info.value)
