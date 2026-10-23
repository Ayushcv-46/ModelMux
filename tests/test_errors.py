from fastapi.testclient import TestClient

from app.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from app.main import app



@app.get("/_test/timeout")
async def _raise_timeout():
    raise ProviderTimeout("provider took too long", provider="mock")


@app.get("/_test/ratelimited")
async def _raise_rate_limited():
    raise ProviderRateLimited("slow down", provider="mock")


@app.get("/_test/error")
async def _raise_error():
    raise ProviderError("provider broke")


client = TestClient(app)


def test_timeout_returns_504():
    response = client.get("/_test/timeout")
    assert response.status_code == 504
    body = response.json()
    assert body["error"]["type"] == "ProviderTimeout"
    assert body["error"]["provider"] == "mock"


def test_rate_limited_returns_429():
    response = client.get("/_test/ratelimited")
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "ProviderRateLimited"


def test_base_error_returns_502():
    response = client.get("/_test/error")
    assert response.status_code == 502
    assert response.json()["error"]["provider"] is None


def test_children_are_provider_errors():
    assert isinstance(ProviderTimeout("x"), ProviderError)
    assert isinstance(ProviderRateLimited("x"), ProviderError)

