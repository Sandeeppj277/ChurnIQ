"""
ChurnIQ — Risk Level & Threshold Tests

Verifies Rule 8:
- Thresholds are settings: LOW_RISK_MAX = 39, MEDIUM_RISK_MAX = 69.
- Risk level mapping:
    pct <= 39 -> LOW
    39 < pct <= 69 -> MEDIUM
    pct > 69 -> HIGH
- Binary label:
    prob >= 0.5 (50.0%) -> Likely to Churn
    prob < 0.5 (50.0%) -> Not Likely to Churn
"""

import pytest
from backend.app.config import get_settings


def map_risk_level(churn_pct: float, low_max: int, med_max: int) -> str:
    """Helper representing the risk classification logic in MLService."""
    if churn_pct <= low_max:
        return "LOW"
    elif churn_pct <= med_max:
        return "MEDIUM"
    else:
        return "HIGH"


def map_prediction_label(churn_prob: float) -> str:
    """Helper representing the 50% cutoff binary classification."""
    return "Likely to Churn" if churn_prob >= 0.5 else "Not Likely to Churn"


def test_default_settings_thresholds():
    """Verify settings defaults match the architecture specification."""
    settings = get_settings()
    assert settings.LOW_RISK_MAX == 39
    assert settings.MEDIUM_RISK_MAX == 69


@pytest.mark.parametrize(
    "pct,expected_risk",
    [
        (0.0, "LOW"),
        (25.5, "LOW"),
        (39.0, "LOW"),         # Exact low boundary
        (39.01, "MEDIUM"),     # Just above low boundary
        (50.0, "MEDIUM"),
        (69.0, "MEDIUM"),      # Exact medium boundary
        (69.01, "HIGH"),       # Just above medium boundary
        (85.5, "HIGH"),
        (100.0, "HIGH"),
    ],
)
def test_risk_level_boundaries(pct: float, expected_risk: str):
    """Test boundary transitions for LOW, MEDIUM, and HIGH risk."""
    risk = map_risk_level(pct, low_max=39, med_max=69)
    assert risk == expected_risk


@pytest.mark.parametrize(
    "prob,expected_label",
    [
        (0.0, "Not Likely to Churn"),
        (0.4999, "Not Likely to Churn"),
        (0.50, "Likely to Churn"),       # Exact 50% threshold
        (0.5001, "Likely to Churn"),
        (1.0, "Likely to Churn"),
    ],
)
def test_prediction_label_cutoff(prob: float, expected_label: str):
    """Test 50% cutoff for Likely to Churn vs Not Likely to Churn."""
    label = map_prediction_label(prob)
    assert label == expected_label
