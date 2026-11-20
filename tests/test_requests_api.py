"""
Day 20 — Request History API & Pagination Unit Tests (GET /v1/requests)

Validates:
1. Filtering by provider (e.g. provider='groq').
2. Filtering by status (e.g. status='ProviderError').
3. Combined filtering (e.g. provider='groq' and status='ProviderTimeout').
4. Pagination: limit and offset correctly slice and paginate results.
5. Newest-first ordering (id descending).
6. Request ID and routing decision persistence.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, RequestLog, init_db
from app.main import get_requests
from app.schemas import RequestLogResponse


@pytest.fixture
def db_session() -> Session:
    """Create an isolated, in-memory SQLite database populated with test logs."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()

    # Seed 6 sample request records
    sample_records = [
        # id 1: groq success
        RequestLog(
            request_id="req_001",
            provider="groq",
            model="openai/gpt-oss-20b",
            status="success",
            latency_ms=120,
            tokens_in=10,
            tokens_out=20,
            routing_decision="PriorityStrategy",
        ),
        # id 2: groq failure (timeout)
        RequestLog(
            request_id="req_002",
            provider="groq",
            model="openai/gpt-oss-20b",
            status="ProviderTimeout",
            latency_ms=5000,
            tokens_in=15,
            tokens_out=0,
            routing_decision="PriorityStrategy",
        ),
        # id 3: gemini success
        RequestLog(
            request_id="req_003",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            status="success",
            latency_ms=300,
            tokens_in=25,
            tokens_out=50,
            routing_decision="RoundRobinStrategy",
        ),
        # id 4: gemini failure (rate limit)
        RequestLog(
            request_id="req_004",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            status="ProviderRateLimited",
            latency_ms=80,
            tokens_in=12,
            tokens_out=0,
            routing_decision="RoundRobinStrategy",
        ),
        # id 5: groq failure (generic error)
        RequestLog(
            request_id="req_005",
            provider="groq",
            model="openai/gpt-oss-120b",
            status="ProviderError",
            latency_ms=450,
            tokens_in=30,
            tokens_out=0,
            routing_decision="ExplicitStrategy",
        ),
        # id 6: ollama success
        RequestLog(
            request_id="req_006",
            provider="ollama",
            model="qwen2.5:3b",
            status="success",
            latency_ms=900,
            tokens_in=40,
            tokens_out=80,
            routing_decision="PriorityStrategy",
        ),
    ]

    for record in sample_records:
        session.add(record)
    session.commit()

    yield session

    session.close()
    Base.metadata.drop_all(bind=test_engine)


# ---------------------------------------------------------------------------
# Filter Tests
# ---------------------------------------------------------------------------

def test_get_all_requests_newest_first(db_session):
    """Calling get_requests with no filter returns all records in descending ID order."""
    results = get_requests(db=db_session)
    assert len(results) == 6
    # Verify newest request is first (id: 6)
    assert results[0].id == 6
    assert results[-1].id == 1


def test_filter_by_provider(db_session):
    """Filter by provider='groq' returns only Groq requests."""
    results = get_requests(provider="groq", db=db_session)
    assert len(results) == 3
    assert all(r.provider == "groq" for r in results)


def test_filter_by_status(db_session):
    """Filter by status='success' returns only successful requests."""
    results = get_requests(status="success", db=db_session)
    assert len(results) == 3
    assert all(r.status == "success" for r in results)


def test_combined_filter_failed_groq_requests(db_session):
    """Filter for failed Groq requests (Day 20 goal: list failed Groq calls)."""
    # Look for timeout failures on Groq
    results = get_requests(provider="groq", status="ProviderTimeout", db=db_session)
    assert len(results) == 1
    assert results[0].request_id == "req_002"
    assert results[0].provider == "groq"
    assert results[0].status == "ProviderTimeout"


def test_case_insensitive_filter(db_session):
    """Filtering handles upper and lowercase gracefully."""
    results = get_requests(provider="GEMINI", status="SUCCESS", db=db_session)
    assert len(results) == 1
    assert results[0].request_id == "req_003"


def test_nonexistent_filter_returns_empty_list(db_session):
    """Querying for a provider with no logs returns an empty list."""
    results = get_requests(provider="unknown_provider", db=db_session)
    assert results == []


# ---------------------------------------------------------------------------
# Pagination Tests
# ---------------------------------------------------------------------------

def test_pagination_limit(db_session):
    """Limit restricts the number of rows returned."""
    results = get_requests(limit=2, db=db_session)
    assert len(results) == 2
    # Should be the two newest items: 6 and 5
    assert results[0].id == 6
    assert results[1].id == 5


def test_pagination_offset(db_session):
    """Offset skips the given number of items."""
    page1 = get_requests(limit=2, offset=0, db=db_session)
    page2 = get_requests(limit=2, offset=2, db=db_session)

    assert len(page1) == 2
    assert len(page2) == 2

    # Verify no overlap between pages
    page1_ids = {r.id for r in page1}
    page2_ids = {r.id for r in page2}
    assert page1_ids.isdisjoint(page2_ids)

    # page1: [6, 5], page2: [4, 3]
    assert [r.id for r in page1] == [6, 5]
    assert [r.id for r in page2] == [4, 3]


def test_response_schema_validation(db_session):
    """Ensure records validate properly against RequestLogResponse schema."""
    results = get_requests(limit=1, db=db_session)
    assert len(results) == 1

    first = results[0]
    validated = RequestLogResponse.model_validate(first)

    assert validated.id == 6
    assert validated.request_id == "req_006"
    assert validated.provider == "ollama"
    assert validated.routing_decision == "PriorityStrategy"
    assert isinstance(validated.created_at, datetime)
