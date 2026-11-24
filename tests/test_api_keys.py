"""
Day 31 — API Keys and Hashing Unit Tests

Verifies:
1. Cryptographic generation of API keys with prefix and entropy.
2. Deterministic SHA-256 hashing (same key -> same hash; different keys -> different hashes).
3. Plaintext keys are NEVER stored in the database table (only 64-char hex hash).
4. Verification flow: valid incoming key matches stored hash and returns ApiKey.
5. Verification failure: invalid, malformed, or altered keys return None.
6. Revocation: setting active=False prevents verification.
7. Uniqueness constraint on key_hash in SQLite.
8. CLI script runs and outputs the secret key once.
"""

import subprocess
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.auth import create_api_key, generate_api_key, hash_api_key, verify_api_key
from app.db import ApiKey, Base, init_db


@pytest.fixture
def auth_db():
    """Create isolated in-memory database session for auth tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    init_db(target_engine=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_generate_api_key_format_and_entropy():
    """Keys should start with 'mm_', have high entropy, and be distinct."""
    key1 = generate_api_key()
    key2 = generate_api_key()

    assert key1.startswith("mm_")
    assert key2.startswith("mm_")
    assert len(key1) >= 40
    assert key1 != key2


def test_hash_api_key_deterministic_sha256():
    """SHA-256 hashing must be deterministic and exactly 64 hex characters."""
    key = "mm_test_secret_key_12345"
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Different key produces different hash
    different_hash = hash_api_key("mm_different_key")
    assert different_hash != hash1


def test_create_api_key_stores_only_hash_in_database(auth_db):
    """
    The Golden Rule:
    The database must ONLY store the SHA-256 hash. The plaintext key
    must not appear anywhere in the database row.
    """
    record, plain_key = create_api_key(
        db=auth_db,
        name="test-client",
        rate_limit=120,
        tenant_id="tenant-alpha",
    )

    assert record.id is not None
    assert record.name == "test-client"
    assert record.rate_limit == 120
    assert record.active is True
    assert record.tenant_id == "tenant-alpha"

    # Hash in DB is 64 hex characters
    assert len(record.key_hash) == 64
    assert record.key_hash == hash_api_key(plain_key)

    # Plain key is NOT stored in any DB column
    for col in ApiKey.__table__.columns:
        val = getattr(record, col.name)
        assert val != plain_key


def test_verify_api_key_success(auth_db):
    """Providing the correct plain key verifies against the stored hash."""
    record, plain_key = create_api_key(auth_db, name="verified-user", rate_limit=50)

    verified = verify_api_key(auth_db, plain_key)
    assert verified is not None
    assert verified.id == record.id
    assert verified.name == "verified-user"
    assert verified.rate_limit == 50


def test_verify_api_key_invalid_and_malformed(auth_db):
    """Supplying a fake, altered, or empty key returns None."""
    create_api_key(auth_db, name="legit-user")

    assert verify_api_key(auth_db, "mm_completely_fake_key") is None
    assert verify_api_key(auth_db, "") is None
    assert verify_api_key(auth_db, "wrong_prefix") is None
    assert verify_api_key(auth_db, None) is None  # type: ignore


def test_verify_api_key_revoked_returns_none(auth_db):
    """A revoked key (active=False) must fail verification even if hash matches."""
    record, plain_key = create_api_key(auth_db, name="revoked-user")

    # Deactivate / revoke key
    record.active = False
    auth_db.commit()

    assert verify_api_key(auth_db, plain_key) is None


def test_api_key_uniqueness_constraint(auth_db):
    """Duplicate key_hash cannot be inserted due to UNIQUE constraint."""
    key = "mm_duplicate_key"
    kh = hash_api_key(key)

    rec1 = ApiKey(name="client1", key_hash=kh)
    auth_db.add(rec1)
    auth_db.commit()

    rec2 = ApiKey(name="client2", key_hash=kh)
    auth_db.add(rec2)
    with pytest.raises(IntegrityError):
        auth_db.commit()


def test_cli_script_generates_and_prints_key():
    """Verify scripts/create_api_key.py can be invoked via CLI and outputs key."""
    cmd = [
        sys.executable,
        "scripts/create_api_key.py",
        "--name",
        "cli-test-client",
        "--rate-limit",
        "80",
        "--tenant",
        "cli-tenant",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    output = result.stdout

    assert "[ModelMux] API KEY CREATED SUCCESSFULLY" in output
    assert "cli-test-client" in output
    assert "cli-tenant" in output
    assert "80 req/min" in output
    assert "YOUR SECRET API KEY:" in output
    assert "mm_" in output
