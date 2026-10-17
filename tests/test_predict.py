"""
ChurnIQ — Prediction & Model Routes Tests

Verifies:
- POST /predict computes churn probability, risk level, and top 5 SHAP factors.
- Customer upsert and prediction logging behavior in the database.
- 422 HTTP validation responses on invalid customer inputs (Rule 6).
- GET /model/metrics meets acceptance targets (Acc > 77%, AUC > 0.80).
- GET /dashboard/summary aggregates prediction counts, risk tiers, and history.
"""

from fastapi.testclient import TestClient


def test_predict_success_and_persistence(client: TestClient, sample_customer_payload: dict):
    """POST /predict computes scores, top factors, and persists prediction to DB."""
    response = client.post("/predict", json=sample_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == sample_customer_payload["customer_id"]
    assert data["prediction"] in ["Likely to Churn", "Not Likely to Churn"]
    assert 0.0 <= data["churn_probability"] <= 100.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["model"] == "XGBoost"
    assert data["prediction_id"] is not None

    # SHAP Factors: exactly 5 items with feature, impact, direction
    assert len(data["top_factors"]) == 5
    for f in data["top_factors"]:
        assert "feature" in f
        assert "impact" in f
        assert f["direction"] in ["increases_risk", "decreases_risk"]
        assert isinstance(f["impact"], float)


def test_repeated_predict_upserts_customer_and_appends_predictions(
    client: TestClient, sample_customer_payload: dict
):
    """POST /predict for the same customer updates profile and logs multiple prediction rows."""
    # First prediction
    client.post("/predict", json=sample_customer_payload)

    # Second prediction with modified tenure
    modified_payload = sample_customer_payload.copy()
    modified_payload["tenure"] = 36
    r2 = client.post("/predict", json=modified_payload)
    assert r2.status_code == 200

    # Dashboard check: total_scored counts prediction instances
    r_dash = client.get("/dashboard/summary")
    assert r_dash.status_code == 200
    dash_data = r_dash.json()
    assert dash_data["total_scored"] >= 2


def test_predict_validation_errors_422(client: TestClient, sample_customer_payload: dict):
    """Invalid types, out-of-range values, or unallowed enums must return HTTP 422 (Rule 6)."""
    # 1. Invalid age > 110
    bad_age = sample_customer_payload.copy()
    bad_age["age"] = 150
    assert client.post("/predict", json=bad_age).status_code == 422

    # 2. Invalid age < 1
    bad_age_zero = sample_customer_payload.copy()
    bad_age_zero["age"] = 0
    assert client.post("/predict", json=bad_age_zero).status_code == 422

    # 3. Invalid payment method enum
    bad_payment = sample_customer_payload.copy()
    bad_payment["payment_method"] = "Gold Coins"
    assert client.post("/predict", json=bad_payment).status_code == 422

    # 4. Negative charges
    bad_charges = sample_customer_payload.copy()
    bad_charges["monthly_charges"] = -10.0
    assert client.post("/predict", json=bad_charges).status_code == 422

    # 5. Invalid customer ID pattern (special characters like spaces/symbols)
    bad_id = sample_customer_payload.copy()
    bad_id["customer_id"] = "ID with spaces and !@#"
    assert client.post("/predict", json=bad_id).status_code == 422


def test_model_metrics_endpoint(client: TestClient):
    """GET /model/metrics serves metrics exceeding acceptance targets."""
    response = client.get("/model/metrics")
    assert response.status_code == 200
    metrics = response.json()

    assert metrics["accuracy"] > 0.77
    assert metrics["roc_auc"] > 0.80
    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    assert all(k in cm for k in ["true_negatives", "false_positives", "false_negatives", "true_positives"])
    assert len(metrics["global_feature_importance"]) == 10


def test_dashboard_summary_endpoint(client: TestClient):
    """GET /dashboard/summary returns total scored, risk breakdown, and model snapshot."""
    response = client.get("/dashboard/summary")
    assert response.status_code == 200
    summary = response.json()

    assert "total_scored" in summary
    assert "risk_counts" in summary
    assert "average_probability" in summary
    assert "recent_predictions" in summary
    assert "model_snapshot" in summary
    assert summary["model_snapshot"]["status"] == "ready"
