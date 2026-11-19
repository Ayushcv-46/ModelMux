import asyncio
from fastapi import Request
import httpx
import pytest

from app.main import app, get_http_client, get_provider, lifespan
from app.providers.groq import GroqProvider


def test_lifespan_initializes_and_closes_http_client():
    """Verify that lifespan startup creates the shared client and shutdown closes it."""
    async def run():
        async with lifespan(app):
            # Assert client was created and stored in app.state
            client = getattr(app.state, "http_client", None)
            assert client is not None
            assert isinstance(client, httpx.AsyncClient)
            assert not client.is_closed

            # Check configured connection pool limits and timeouts
            pool = client._transport._pool
            assert pool._max_keepalive_connections == 20
            assert pool._max_connections == 100
            assert client.timeout.connect == 5.0

        # After exiting lifespan context, assert the client was closed gracefully
        assert client.is_closed

    asyncio.run(run())


def test_get_http_client_dependency():
    """Verify get_http_client extracts client from request.app.state."""
    dummy_client = httpx.AsyncClient()

    class DummyState:
        http_client = dummy_client

    class DummyApp:
        state = DummyState()

    class DummyRequest:
        app = DummyApp()

    client = get_http_client(DummyRequest())
    assert client is dummy_client


def test_get_provider_receives_shared_client():
    """Verify get_provider injects the shared HTTP client into GroqProvider."""
    dummy_client = httpx.AsyncClient()
    provider = get_provider(client=dummy_client)

    assert isinstance(provider, GroqProvider)
    assert provider._client is dummy_client
