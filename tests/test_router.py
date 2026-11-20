"""
Day 18 & 19 — Router & Strategy Unit Tests

Validates:
1. PriorityStrategy: respects configured ordering and skips missing providers.
2. RoundRobinStrategy: cycles through providers sequentially on each request.
3. ExplicitStrategy: directs requests to the exact provider matching the model.
4. Model compatibility: rejects unsupported models with ModelNotSupportedError.
5. YAML config-driven loading: parses strategies and model registries from config.
"""

import pytest

from app.config import AppConfig, ProviderConfig, RoutingConfig, load_yaml_config
from app.errors import ModelNotSupportedError, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.routing.router import Router
from app.routing.strategy import (
    ExplicitStrategy,
    PriorityStrategy,
    RoundRobinStrategy,
)
from app.schemas import ChatRequest, Message


@pytest.fixture
def sample_providers():
    return {
        "groq": GroqProvider(api_key="fake-groq-key"),
        "gemini": GeminiProvider(api_key="fake-gemini-key"),
        "ollama": OllamaProvider(),
        "mock": MockProvider(),
    }


@pytest.fixture
def sample_model_mapping():
    return {
        "openai/gpt-oss-20b": "groq",
        "openai/gpt-oss-120b": "groq",
        "gemini-3.5-flash-lite": "gemini",
        "qwen2.5:3b": "ollama",
        "mock-1": "mock",
    }


def make_request(model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        model=model,
    )


# ---------------------------------------------------------------------------
# PriorityStrategy Tests
# ---------------------------------------------------------------------------

def test_priority_strategy_respects_custom_order(sample_providers):
    """PriorityStrategy orders providers matching the input priority list."""
    strategy = PriorityStrategy(priority=["gemini", "ollama", "groq"])
    router = Router(providers=sample_providers, strategy=strategy)

    ordered = router.get_providers()
    names = [p.name for p in ordered]

    assert names == ["gemini", "ollama", "groq"]
    assert router.get_primary_provider().name == "gemini"


def test_changing_priority_changes_primary_provider(sample_providers):
    """Switching priority list from groq-first to ollama-first changes the primary provider."""
    router1 = Router(
        providers=sample_providers,
        strategy=PriorityStrategy(priority=["groq", "gemini", "ollama"]),
    )
    assert router1.get_primary_provider().name == "groq"

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


# ---------------------------------------------------------------------------
# RoundRobinStrategy Tests (Day 19)
# ---------------------------------------------------------------------------

def test_round_robin_strategy_cycles_providers(sample_providers):
    """RoundRobinStrategy rotates through providers sequentially with each invocation."""
    order = ["groq", "gemini", "ollama"]
    strategy = RoundRobinStrategy(provider_order=order)
    router = Router(providers=sample_providers, strategy=strategy)

    # 1st call -> groq first
    res1 = router.get_providers()
    assert [p.name for p in res1] == ["groq", "gemini", "ollama"]
    assert router.get_primary_provider().name == "gemini"  # Note: next call advances index

    # 3rd call -> ollama first
    res3 = router.get_providers()
    assert [p.name for p in res3] == ["ollama", "groq", "gemini"]

    # 4th call -> cycles back to groq first
    res4 = router.get_providers()
    assert [p.name for p in res4] == ["groq", "gemini", "ollama"]


def test_round_robin_defaults_to_all_providers_when_no_order_given(sample_providers):
    """Without an explicit order list, round robin cycles all available providers."""
    strategy = RoundRobinStrategy()
    router = Router(providers=sample_providers, strategy=strategy)

    first_p = router.get_primary_provider()
    second_p = router.get_primary_provider()

    assert first_p != second_p


# ---------------------------------------------------------------------------
# ExplicitStrategy Tests (Day 19)
# ---------------------------------------------------------------------------

def test_explicit_strategy_routes_to_exact_provider(sample_providers, sample_model_mapping):
    """ExplicitStrategy selects the provider mapped to the client's requested model."""
    strategy = ExplicitStrategy(model_to_provider=sample_model_mapping)
    router = Router(providers=sample_providers, strategy=strategy)

    # Ask for groq's model
    req_groq = make_request("openai/gpt-oss-20b")
    assert router.get_primary_provider(req_groq).name == "groq"

    # Ask for gemini's model
    req_gemini = make_request("gemini-3.5-flash-lite")
    assert router.get_primary_provider(req_gemini).name == "gemini"

    # Ask for ollama's model
    req_ollama = make_request("qwen2.5:3b")
    assert router.get_primary_provider(req_ollama).name == "ollama"


def test_explicit_strategy_rejects_incompatible_model(sample_providers, sample_model_mapping):
    """Requesting an unsupported model raises ModelNotSupportedError (HTTP 400)."""
    strategy = ExplicitStrategy(model_to_provider=sample_model_mapping)
    router = Router(providers=sample_providers, strategy=strategy)

    req_invalid = make_request("unsupported-claude-9")
    with pytest.raises(ModelNotSupportedError) as exc_info:
        router.get_providers(req_invalid)

    assert "not supported by any registered provider" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_explicit_strategy_falls_back_when_model_is_none(sample_providers, sample_model_mapping):
    """When no model is requested, ExplicitStrategy delegates to default strategy."""
    fallback = PriorityStrategy(priority=["ollama", "groq", "gemini"])
    strategy = ExplicitStrategy(
        model_to_provider=sample_model_mapping,
        default_strategy=fallback,
    )
    router = Router(providers=sample_providers, strategy=strategy)

    req_no_model = make_request(model=None)
    primary = router.get_primary_provider(req_no_model)
    assert primary.name == "ollama"


# ---------------------------------------------------------------------------
# Config-Driven Design Tests (Day 19)
# ---------------------------------------------------------------------------

def test_load_yaml_config_parses_file(tmp_path):
    """Verify load_yaml_config correctly parses strategies and provider models."""
    sample_yaml = """
routing:
  strategy: "round_robin"
  priority:
    - "gemini"
    - "groq"

providers:
  gemini:
    models:
      - "gemini-1.5-flash"
  groq:
    models:
      - "llama-3.1-8b-instant"
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(sample_yaml, encoding="utf-8")

    cfg = load_yaml_config(cfg_file)
    assert cfg.routing.strategy == "round_robin"
    assert cfg.routing.priority == ["gemini", "groq"]
    assert cfg.model_to_provider["gemini-1.5-flash"] == "gemini"
    assert cfg.model_to_provider["llama-3.1-8b-instant"] == "groq"
