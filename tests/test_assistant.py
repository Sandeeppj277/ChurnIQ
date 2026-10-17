"""
ChurnIQ — Grounded Assistant Chat Tests

Verifies:
- POST /ai/chat provides grounded responses using stored customer context.
- Gracefully falls back to deterministic grounded answering when offline (Rules 4 & 10).
- Labels output source as 'rule_based'.
- Handles unknown customers (404) and invalid queries (422).
"""

from fastapi.testclient import TestClient


def test_assistant_chat_rule_based_grounding(client: TestClient, sample_customer_payload: dict):
    """POST /ai/chat returns grounded factual answer with source='rule_based' when offline."""
    client.post("/predict", json=sample_customer_payload)
    cid = sample_customer_payload["customer_id"]

    # Ask about risk
    r = client.post(
        "/ai/chat",
        json={"customer_id": cid, "question": "Why is this customer at risk?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["customer_id"] == cid
    assert data["source"] == "rule_based"
    assert "risk" in data["answer"].lower()
    assert data["error"] is not None

    # Ask about contract
    r_contract = client.post(
        "/ai/chat",
        json={"customer_id": cid, "question": "What is their contract type?"},
    )
    assert r_contract.status_code == 200
    assert "One-Year" in r_contract.json()["answer"]


def test_assistant_chat_404_for_unknown_customer(client: TestClient):
    """Chat endpoint returns 404 for non-existent customer."""
    r = client.post("/ai/chat", json={"customer_id": "GHOST_000", "question": "Hello"})
    assert r.status_code == 404


def test_assistant_chat_422_empty_question(client: TestClient, sample_customer_payload: dict):
    """Chat endpoint rejects empty question with HTTP 422."""
    client.post("/predict", json=sample_customer_payload)
    r = client.post(
        "/ai/chat",
        json={"customer_id": sample_customer_payload["customer_id"], "question": ""},
    )
    assert r.status_code == 422
