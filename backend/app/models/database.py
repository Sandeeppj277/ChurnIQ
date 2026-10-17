"""
ChurnIQ — Database Engine & Session Configuration

Supports both SQLite (default) and PostgreSQL (via DATABASE_URL).
Provides the SQLAlchemy Base, SessionLocal factory, get_db FastAPI dependency,
and table initialization helper.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.app.config import get_settings

settings = get_settings()

db_url = settings.DATABASE_URL

# SQLite specific connect args
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables defined in the metadata."""
    import backend.app.models.tables  # noqa: F401 - ensure models are imported
    Base.metadata.create_all(bind=engine)
