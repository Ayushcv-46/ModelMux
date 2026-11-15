import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, RequestLog, get_db, init_db
from app.main import app, get_provider
from app.providers.mock import MockProvider


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database that persists across connections."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Keeps the exact same in-memory DB alive across connections
    )
    init_db(target_engine=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield session

    session.close()
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_provider, None)


def test_chat_logs_three_calls_to_database(test_db_session):
    client = TestClient(app)

    # 1. Call: Success with MockProvider
    app.dependency_overrides[get_provider] = lambda: MockProvider(min_delay=0.0, max_delay=0.0)
    res1 = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello one"}]})
    assert res1.status_code == 200

    # 2. Call: Another Success
    res2 = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello two"}]})
    assert res2.status_code == 200

    # 3. Call: Failure with forced timeout
    app.dependency_overrides[get_provider] = lambda: MockProvider(
        failure_rate=1.0,
        error_kind="timeout",
        min_delay=0.0,
        max_delay=0.0,
    )
    res3 = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello three"}]})
    assert res3.status_code == 504

    # Verification: Assert exactly 3 rows were written in the database
    rows = test_db_session.query(RequestLog).all()
    assert len(rows) == 3

    # Check the first two calls logged as 'success'
    assert rows[0].status == "success"
    assert rows[0].tokens_in > 0
    assert rows[0].tokens_out > 0

    assert rows[1].status == "success"

    # Check the third call logged as 'ProviderTimeout'
    assert rows[2].status == "ProviderTimeout"
    assert rows[2].tokens_out == 0