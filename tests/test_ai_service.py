"""
ChurnIQ — AI Service & Fallback Tests

Verifies Rules 4, 5 & 10:
- Tests never touch the network; GEMINI_API_KEY is cleared.
- AI failure or missing key falls back smoothly to rule-based generation with HTTP 200.
- Every output is explicitly labeled as source='rule_based'.
- Tone-adjustment and 404 handling.
"""

from fastapi.testclient import TestClient
import pytest


def test_ai_recommendation_rule_based_fallback(client: TestClient, high_risk_customer_payload: dict):
    """POST /ai/recommendation generates structured advice with source='rule_based' when offline."""
    # Score customer first
    pred_res = client.post("/predict", json=high_risk_customer_payload)
    assert pred_res.status_code == 200
    cid = high_risk_customer_payload["customer_id"]

    # Request recommendation without network key
    rec_res = client.post("/ai/recommendation", json={"customer_id": cid})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()

    assert rec_data["customer_id"] == cid
    assert rec_data["source"] == "rule_based"
    assert rec_data["priority"] in ["High", "Medium", "Low"]
    assert len(rec_data["explanation"]) > 10
    assert len(rec_data["recommended_action"]) > 10
    assert len(rec_data["reason"]) > 10
    assert rec_data["error"] is not None  # Notice explaining fallback


@pytest.mark.parametrize("tone", ["Professional", "Friendly", "Empathetic", "Short"])
def test_ai_message_rule_based_tones(
    client: TestClient, high_risk_customer_payload: dict, tone: str
):
    """POST /ai/message produces tone-specific outreach labeled source='rule_based'."""
    client.post("/predict", json=high_risk_customer_payload)
    cid = high_risk_customer_payload["customer_id"]

    msg_res = client.post("/ai/message", json={"customer_id": cid, "tone": tone})
    assert msg_res.status_code == 200
    msg_data = msg_res.json()

    assert msg_data["customer_id"] == cid
    assert msg_data["tone"] == tone
    assert msg_data["source"] == "rule_based"
    assert len(msg_data["message"]) > 15


def test_ai_endpoints_404_for_unknown_customer(client: TestClient):
    """AI endpoints return HTTP 404 if the customer does not exist in the database."""
    r_rec = client.post("/ai/recommendation", json={"customer_id": "UNKNOWN_999"})
    assert r_rec.status_code == 404

    r_msg = client.post("/ai/message", json={"customer_id": "UNKNOWN_999", "tone": "Friendly"})
    assert r_msg.status_code == 404


def test_ai_message_422_on_invalid_tone(client: TestClient, sample_customer_payload: dict):
    """POST /ai/message returns 422 for unrecognized tone values."""
    client.post("/predict", json=sample_customer_payload)
    r = client.post(
        "/ai/message",
        json={"customer_id": sample_customer_payload["customer_id"], "tone": "Sarcastic"},
    )
    assert r.status_code == 422
