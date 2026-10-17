"""
ChurnIQ — Test Configuration & Fixtures

Enforces Rule 10: Tests never touch the network.
Clears GEMINI_API_KEY in the test configuration and provides:
- Isolated in-memory SQLite database
- FastAPI TestClient with database dependency override
- Standard sample customer payloads
"""

import os
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Ensure tests run with network disabled for Gemini (Rule 10)
os.environ["GEMINI_API_KEY"] = ""

from backend.app.config import get_settings
get_settings.cache_clear()

from backend.app.models.database import Base, get_db
from api.main import app

# In-memory SQLite database for isolated test execution
TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all database tables for the test session."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a transactional test database session."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_customer_payload() -> dict:
    """Standard valid customer payload."""
    return {
        "customer_id": "TEST_C100",
        "customer_name": "Test Customer",
        "gender": "Female",
        "age": 36,
        "tenure": 14,
        "subscription_type": "Standard",
        "contract_type": "One-Year",
        "monthly_charges": 60.0,
        "total_charges": 840.0,
        "payment_method": "Credit Card",
        "payment_status": "On Time",
        "usage_frequency": 16.0,
        "service_usage": 60.0,
        "complaint_count": 0,
        "support_calls": 1,
        "engagement_score": 68.0,
    }


@pytest.fixture
def high_risk_customer_payload() -> dict:
    """Payload representing a high churn risk customer."""
    return {
        "customer_id": "TEST_C999",
        "customer_name": "High Risk Customer",
        "gender": "Male",
        "age": 45,
        "tenure": 2,
        "subscription_type": "Basic",
        "contract_type": "Month-to-Month",
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "payment_method": "Credit Card",
        "payment_status": "Failed",
        "usage_frequency": 3.0,
        "service_usage": 15.0,
        "complaint_count": 3,
        "support_calls": 5,
        "engagement_score": 12.0,
    }
