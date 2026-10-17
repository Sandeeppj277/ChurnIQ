"""
ChurnIQ — Health Endpoint Tests

Verifies GET /health endpoint returns HTTP 200, system status, model readiness,
and AI configuration flag.
"""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """GET /health must return 200 with status, model_ready, and ai_enabled."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_ready" in data
    assert "ai_enabled" in data
    assert isinstance(data["model_ready"], bool)
    assert isinstance(data["ai_enabled"], bool)
