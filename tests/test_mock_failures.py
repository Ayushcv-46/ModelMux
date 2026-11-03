import pytest

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, Message


def make_request():
    return ChatRequest(messages=[Message(role="user", content="hello there")])


def fast_mock(**kwargs):
    return MockProvider(min_delay=0, max_delay=0, **kwargs)


async def outcomes(provider, n=20):
    """Run n calls and record 'ok' or the error class name for each."""
    results = []
    for _ in range(n):
        try:
            await provider.complete(make_request())
            results.append("ok")
        except Exception as exc:
            results.append(type(exc).__name__)
    return results


async def test_rate_zero_never_fails():
    provider = fast_mock(failure_rate=0.0)
    assert set(await outcomes(provider, 50)) == {"ok"}


async def test_rate_one_always_fails():
    provider = fast_mock(failure_rate=1.0)
    assert "ok" not in await outcomes(provider, 50)


@pytest.mark.parametrize(
    "kind, expected",
    [("500", ProviderError), ("timeout", ProviderTimeout), ("429", ProviderRateLimited)],
)
async def test_error_kind_raises_matching_error(kind, expected):
    provider = fast_mock(failure_rate=1.0, error_kind=kind)
    with pytest.raises(expected):
        await provider.complete(make_request())


async def test_same_seed_gives_same_sequence():
    a = fast_mock(failure_rate=0.5, seed=42)
    b = fast_mock(failure_rate=0.5, seed=42)
    first, second = await outcomes(a), await outcomes(b)
    assert first == second
    assert len(set(first)) == 2  # both successes and failures happened


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        MockProvider(failure_rate=1.5)
    with pytest.raises(ValueError):
        MockProvider(error_kind="timeot")