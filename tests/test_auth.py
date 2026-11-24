"""
Day 32 — Auth Dependency & Admin Routes Unit Tests

Verifies:
1. Missing Authorization header returns HTTP 401 Unauthorized.
2. Unknown or invalid API key returns HTTP 401 Unauthorized.
3. Revoked API key (active=False) returns HTTP 403 Forbidden.
4. Valid active API key allows requests and logs tenant_id (200 OK).
5. POST /admin/keys creates a key when valid ADMIN_SECRET is provided.
6. POST /admin/keys returns 401 when ADMIN_SECRET is missing or wrong.
7. POST /admin/keys/revoke deactivates a key, which subsequently returns 403 on /chat.
8. Tenant Scoping: GET /v1/requests filters results to the authenticated tenant's data only.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import create_api_key, hash_api_key
from app.db import ApiKey, Base, RequestLog, get_db, init_db
from app.main import app, get_provider
from app.providers.mock import MockProvider


@pytest.fixture
def auth_test_db():
    """Create isolated in-memory database with clean schema for auth tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_provider] = lambda: MockProvider(min_delay=0.0, max_delay=0.0)

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_provider, None)


# ---------------------------------------------------------------------------
# Core Auth Dependency Tests (401 vs 403)
# ---------------------------------------------------------------------------

def test_chat_missing_api_key_returns_401(auth_test_db):
    """Calling /chat with no Authorization header must return HTTP 401."""
    client = TestClient(app)
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


def test_chat_unknown_api_key_returns_401(auth_test_db):
    """Calling /chat with an unknown/fabricated key must return HTTP 401."""
    client = TestClient(app)
    response = client.post(
        "/chat",
        headers={"Authorization": "Bearer mm_totally_fake_random_key_123"},
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_chat_revoked_api_key_returns_403(auth_test_db):
    """Calling /chat with a revoked key (active=False) must return HTTP 403 Forbidden."""
    record, plain_key = create_api_key(auth_test_db, name="banned-tenant", tenant_id="tenant-banned")

    # Deactivate / revoke key
    record.active = False
    auth_test_db.commit()

    client = TestClient(app)
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {plain_key}"},
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 403
    assert "revoked" in response.json()["detail"].lower()


def test_chat_valid_api_key_returns_200_and_logs_tenant(auth_test_db):
    """Calling /chat with a valid active key succeeds and logs tenant_id in SQLite."""
    record, plain_key = create_api_key(auth_test_db, name="acme-corp", tenant_id="tenant-acme-123")

    client = TestClient(app)
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {plain_key}"},
        json={"messages": [{"role": "user", "content": "Hello ModelMux"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data

    # Verify that the database RequestLog records the tenant_id
    log = auth_test_db.query(RequestLog).order_by(RequestLog.id.desc()).first()
    assert log is not None
    assert log.tenant_id == "tenant-acme-123"


# ---------------------------------------------------------------------------
# Admin Routes Tests (ADMIN_SECRET)
# ---------------------------------------------------------------------------

def test_admin_create_key_success(auth_test_db):
    """Admin can create a key by providing the valid ADMIN_SECRET."""
    client = TestClient(app)
    response = client.post(
        "/admin/keys",
        headers={"X-Admin-Secret": "modelmux-admin-secret-dev"},
        json={"name": "new-partner", "rate_limit": 120, "tenant_id": "tenant-partner"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-partner"
    assert data["rate_limit"] == 120
    assert data["tenant_id"] == "tenant-partner"
    assert data["key"].startswith("mm_")
    assert data["active"] is True

    # The newly created key should work immediately on /chat
    plain_key = data["key"]
    chat_resp = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {plain_key}"},
        json={"messages": [{"role": "user", "content": "Testing new key"}]},
    )
    assert chat_resp.status_code == 200


def test_admin_create_key_unauthorized_without_secret(auth_test_db):
    """Admin endpoints reject requests missing or using incorrect ADMIN_SECRET with 401."""
    client = TestClient(app)

    # Missing header
    res1 = client.post("/admin/keys", json={"name": "hacker", "rate_limit": 1000})
    assert res1.status_code == 401

    # Wrong secret
    res2 = client.post(
        "/admin/keys",
        headers={"X-Admin-Secret": "wrong-secret"},
        json={"name": "hacker", "rate_limit": 1000},
    )
    assert res2.status_code == 401


def test_admin_revoke_key_flow(auth_test_db):
    """Admin can revoke a key, preventing future access (returns 403)."""
    record, plain_key = create_api_key(auth_test_db, name="client-to-revoke")

    client = TestClient(app)

    # Revoke via key_id
    revoke_resp = client.post(
        "/admin/keys/revoke",
        headers={"X-Admin-Secret": "modelmux-admin-secret-dev"},
        json={"key_id": record.id},
    )
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["status"] == "revoked"

    # Now calling /chat with this key should return 403 Forbidden!
    chat_resp = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {plain_key}"},
        json={"messages": [{"role": "user", "content": "Should fail"}]},
    )
    assert chat_resp.status_code == 403


# ---------------------------------------------------------------------------
# Tenant Scoping Tests (Engineering Refinement)
# ---------------------------------------------------------------------------

def test_tenant_scoping_request_history(auth_test_db):
    """
    Engineering Refinement:
    Client A with Tenant A's key only sees Tenant A's request logs.
    Client B with Tenant B's key only sees Tenant B's request logs.
    """
    _, key_a = create_api_key(auth_test_db, name="tenant-a-key", tenant_id="tenant-alpha")
    _, key_b = create_api_key(auth_test_db, name="tenant-b-key", tenant_id="tenant-beta")

    client = TestClient(app)

    # 1. Tenant A makes 2 requests
    client.post("/chat", headers={"Authorization": f"Bearer {key_a}"}, json={"messages": [{"role": "user", "content": "A1"}]})
    client.post("/chat", headers={"Authorization": f"Bearer {key_a}"}, json={"messages": [{"role": "user", "content": "A2"}]})

    # 2. Tenant B makes 1 request
    client.post("/chat", headers={"Authorization": f"Bearer {key_b}"}, json={"messages": [{"role": "user", "content": "B1"}]})

    # 3. Tenant A queries /v1/requests
    res_a = client.get("/v1/requests", headers={"Authorization": f"Bearer {key_a}"})
    assert res_a.status_code == 200
    logs_a = res_a.json()
    assert len(logs_a) == 2
    assert all(item["tenant_id"] == "tenant-alpha" for item in logs_a)

    # 4. Tenant B queries /v1/requests
    res_b = client.get("/v1/requests", headers={"Authorization": f"Bearer {key_b}"})
    assert res_b.status_code == 200
    logs_b = res_b.json()
    assert len(logs_b) == 1
    assert logs_b[0]["tenant_id"] == "tenant-beta"
