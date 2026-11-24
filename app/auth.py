"""
Authentication and API Key Management for ModelMux (Days 31 & 32).

Implements:
- Cryptographically random key generation (secrets.token_urlsafe).
- SHA-256 one-way deterministic hashing.
- FastAPI Security Dependency (get_current_api_key):
  - 401 Unauthorized for missing or unknown keys.
  - 403 Forbidden for revoked keys (active=False).
- Admin authorization dependency (verify_admin_secret) using ADMIN_SECRET from .env.
- Constant-time hash verification via hmac.compare_digest.
"""

import hashlib
import hmac
import secrets
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db import ApiKey, get_db


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

    if hmac.compare_digest(record.key_hash, incoming_hash):
        return record

    return None


def extract_api_key(authorization: str | None = None, x_api_key: str | None = None) -> str | None:
    """
    Extract plaintext API key from Authorization header or X-API-Key header.
    Supports formats:
    - Authorization: Bearer <key>
    - Authorization: <key>
    - X-API-Key: <key>
    """
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()

    if not authorization or not authorization.strip():
        return None

    parts = authorization.strip().split()
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return parts[-1]


async def get_current_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> ApiKey:
    """
    FastAPI security dependency to authenticate requests via API Key.

    Raises:
        HTTPException(401): If key is missing or unknown.
        HTTPException(403): If key exists in database but has been revoked (active=False).
    Returns:
        ApiKey: Authenticated active API key database record.
    """
    plain_key = extract_api_key(authorization=authorization, x_api_key=x_api_key)
    if not plain_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key in Authorization or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    incoming_hash = hash_api_key(plain_key)
    record = db.query(ApiKey).filter(ApiKey.key_hash == incoming_hash).first()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not record.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has been revoked.",
        )

    return record


async def get_optional_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> ApiKey | None:
    """
    Optional API key dependency for routes that can be public or tenant-scoped.
    Returns ApiKey if valid key provided, None if unauthenticated, or raises 401/403 on invalid.
    """
    plain_key = extract_api_key(authorization=authorization, x_api_key=x_api_key)
    if not plain_key:
        return None

    incoming_hash = hash_api_key(plain_key)
    record = db.query(ApiKey).filter(ApiKey.key_hash == incoming_hash).first()

    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    if not record.active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key has been revoked.")

    return record


async def verify_admin_secret(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret"),
) -> str:
    """
    FastAPI security dependency to protect /admin routes using ADMIN_SECRET.

    Accepts secret via:
    - X-Admin-Secret: <secret>
    - Authorization: Bearer <secret>
    """
    provided_secret = None
    if x_admin_secret and x_admin_secret.strip():
        provided_secret = x_admin_secret.strip()
    elif authorization and authorization.strip():
        parts = authorization.strip().split()
        provided_secret = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else parts[0]

    admin_secret = getattr(settings, "admin_secret", "") or "modelmux-admin-secret-dev"

    if not provided_secret or not hmac.compare_digest(provided_secret, admin_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin secret.",
        )

    return provided_secret
