import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, RequestLog, init_db


@pytest.fixture
def in_memory_db():
    """Create a temporary, in-memory SQLite database for isolated testing."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    init_db(target_engine=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


def test_insert_and_read_request_log(in_memory_db):
    """Verify that a RequestLog row can be inserted and queried back correctly."""
    log_entry = RequestLog(
        provider="mock",
        model="mock-v1",
        status="success",
        latency_ms=120,
        tokens_in=15,
        tokens_out=42,
    )

    in_memory_db.add(log_entry)
    in_memory_db.commit()
    in_memory_db.refresh(log_entry)

    assert log_entry.id is not None
    assert log_entry.id > 0

    # Read back from database
    fetched = in_memory_db.query(RequestLog).filter_by(id=log_entry.id).first()
    assert fetched is not None
    assert fetched.provider == "mock"
    assert fetched.model == "mock-v1"
    assert fetched.status == "success"
    assert fetched.latency_ms == 120
    assert fetched.tokens_in == 15
    assert fetched.tokens_out == 42
    assert fetched.created_at is not None