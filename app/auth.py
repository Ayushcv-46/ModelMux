"""
Authentication and API Key Management for ModelMux (Day 31).

Implements the Golden Rule of Key Security:
- Keys are cryptographically random strings generated via secrets.token_urlsafe.
- Only SHA-256 hashes are persisted in the database.
- The plain key is displayed exactly once upon creation and never stored.
- Incoming keys are hashed and matched against stored hashes in constant-time.
"""

import hashlib
import hmac
import secrets
from sqlalchemy.orm import Session

from app.db import ApiKey


def generate_api_key(prefix: str = "mm_") -> str:
    """
    Generate a cryptographically secure random API key.
    
    Format: mm_<43-character-urlsafe-base64-token> (256 bits of entropy)
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}{random_part}"


def hash_api_key(plain_key: str) -> str:
    """
    Compute deterministic SHA-256 hexadecimal hash of an API key.
    
    Output is always 64 hexadecimal characters.
    """
    return hashlib.sha256(plain_key.strip().encode("utf-8")).hexdigest()


def create_api_key(
    db: Session,
    name: str,
    rate_limit: int = 60,
    tenant_id: str | None = None,
) -> tuple[ApiKey, str]:
    """
    Generate a new API key, store its SHA-256 hash in SQLite, and return both
    the ApiKey record and the plain key.

    CRITICAL SECURITY CONTRACT:
    The returned `plain_key` string must be presented to the user ONCE.
    It will never be stored in the database or logs.
    """
    plain_key = generate_api_key()
    key_hash = hash_api_key(plain_key)

    api_key_record = ApiKey(
        name=name.strip(),
        key_hash=key_hash,
        rate_limit=rate_limit,
        active=True,
        tenant_id=tenant_id.strip() if tenant_id else name.strip(),
    )

    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)

    return api_key_record, plain_key


def verify_api_key(db: Session, plain_key: str) -> ApiKey | None:
    """
    Verify an incoming plaintext API key against the database:
    1. Compute SHA-256 hash of the incoming key.
    2. Look up the active ApiKey row by key_hash in SQLite.
    3. Perform constant-time comparison to prevent timing attacks.
    4. Return the ApiKey record if valid and active, else None.
    """
    if not plain_key or not isinstance(plain_key, str):
        return None

    incoming_hash = hash_api_key(plain_key)
    record = db.query(ApiKey).filter(ApiKey.key_hash == incoming_hash, ApiKey.active.is_(True)).first()

    if record is None:
        return None

    # Constant-time comparison prevents timing side-channel attacks
    if hmac.compare_digest(record.key_hash, incoming_hash):
        return record

    return None
