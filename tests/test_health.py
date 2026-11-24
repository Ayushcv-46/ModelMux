from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_and_redis_fields():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "redis" in data
    assert data["redis"] in ("up", "down")
