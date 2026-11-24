from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

DATABASE_URL = "sqlite:///./modelmux.db"

# connect_args={"check_same_thread": False} is required for SQLite when accessed by multi-threaded FastAPI workers
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    routing_decision: Mapped[str | None] = mapped_column(String(50), nullable=True)
    attempts: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True, default=list)
    tenant_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)


class ApiKey(Base):
    """
    API Keys for ModelMux clients (Day 31).

    Stores ONLY the SHA-256 hash of the API key, never the plain key.
    Includes tenant identification and per-key rate limits.
    """
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    rate_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )


def init_db(target_engine=None) -> None:
    """Create all tables in the target database and ensure columns exist."""
    bind = target_engine or engine
    Base.metadata.create_all(bind=bind)

    with bind.connect() as conn:
        inspector = inspect(bind)
        if "requests" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("requests")]
            if "request_id" not in columns:
                conn.execute(text("ALTER TABLE requests ADD COLUMN request_id VARCHAR(50)"))
            if "routing_decision" not in columns:
                conn.execute(text("ALTER TABLE requests ADD COLUMN routing_decision VARCHAR(50)"))
            if "attempts" not in columns:
                conn.execute(text("ALTER TABLE requests ADD COLUMN attempts JSON"))
            if "tenant_id" not in columns:
                conn.execute(text("ALTER TABLE requests ADD COLUMN tenant_id VARCHAR(50)"))
            conn.commit()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure default database schema and column migrations are applied on import
init_db()
