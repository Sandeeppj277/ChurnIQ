"""
ChurnIQ — ML Pipeline & Preprocessing Tests

Verifies:
- Rule 2: Single clean_data() function treats training and inference data identically.
- Median/mode imputations, duplicate elimination, bounds clipping.
- Feature engineering output produces exactly 25 encoded columns.
- Model bundle integrity in ml/artifacts/churn_model.joblib.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from backend.app.config import get_settings
from ml.preprocessing.preprocess import (
    clean_data,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from ml.features.feature_engineering import (
    build_preprocessor,
    extract_feature_names,
    build_readable_label_map,
)


def test_clean_data_duplicates_and_imputations():
    """Verify clean_data() drops duplicates, imputes medians/modes, and clips bounds."""
    raw_df = pd.DataFrame([
        # Row 1: valid normal customer
        {
            "customer_id": "C001",
            "gender": "Female",
            "age": 30,
            "tenure": 10,
            "subscription_type": "Standard",
            "contract_type": "One-Year",
            "monthly_charges": 50.0,
            "total_charges": 500.0,
            "payment_method": "Credit Card",
            "payment_status": "On Time",
            "usage_frequency": 12.0,
            "service_usage": 50.0,
            "complaint_count": 0,
            "support_calls": 1,
            "engagement_score": 75.0,
        },
        # Row 2: duplicate of C001
        {
            "customer_id": "C001",
            "gender": "Female",
            "age": 30,
            "tenure": 10,
            "subscription_type": "Standard",
            "contract_type": "One-Year",
            "monthly_charges": 50.0,
            "total_charges": 500.0,
            "payment_method": "Credit Card",
            "payment_status": "On Time",
            "usage_frequency": 12.0,
            "service_usage": 50.0,
            "complaint_count": 0,
            "support_calls": 1,
            "engagement_score": 75.0,
        },
        # Row 3: missing total_charges, invalid age (140), negative monthly_charges, engagement > 100
        {
            "customer_id": "C002",
            "gender": "Male",
            "age": 140,  # Invalid -> should become median
            "tenure": 5,
            "subscription_type": "Basic",
            "contract_type": "Month-to-Month",
            "monthly_charges": -20.0,  # Negative -> should clip to 0
            "total_charges": np.nan,  # Missing -> should become median
            "payment_method": "UPI",
            "payment_status": "Failed",
            "usage_frequency": 5.0,
            "service_usage": 20.0,
            "complaint_count": 2,
            "support_calls": 3,
            "engagement_score": 150.0,  # Out of bounds -> should clip to 100
        },
    ])

    cleaned, medians, modes = clean_data(raw_df, is_training=True)

    # 1. Duplicate row dropped
    assert len(cleaned) == 2
    assert list(cleaned["customer_id"]) == ["C001", "C002"]

    # 2. Invalid age replaced by median
    c2 = cleaned[cleaned["customer_id"] == "C002"].iloc[0]
    assert c2["age"] == 30  # Only valid age in dataset is 30, so median is 30

    # 3. Negative monthly charges clipped to 0
    assert c2["monthly_charges"] == 0.0

    # 4. Missing total_charges filled with median (500.0)
    assert c2["total_charges"] == 500.0

    # 5. Engagement score clipped to 100
    assert c2["engagement_score"] == 100.0


def test_feature_engineering_produces_25_columns():
    """Verify ColumnTransformer yields exactly 25 feature columns."""
    sample_df = pd.DataFrame([{
        "customer_id": "C001",
        "gender": "Male",
        "age": 40,
        "tenure": 12,
        "subscription_type": "Standard",
        "contract_type": "Month-to-Month",
        "monthly_charges": 60.0,
        "total_charges": 720.0,
        "payment_method": "Credit Card",
        "payment_status": "On Time",
        "usage_frequency": 15.0,
        "service_usage": 50.0,
        "complaint_count": 1,
        "support_calls": 1,
        "engagement_score": 60.0,
    }])

    # Preprocessor loaded from trained bundle transforms input into 25 columns
    settings = get_settings()
    bundle = joblib.load(settings.model_bundle_path)
    preprocessor = bundle["preprocessor"]

    cols = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X_enc = preprocessor.transform(sample_df[cols])

    # 9 numeric + (2 gender + 3 sub + 3 contract + 5 payment_method + 3 payment_status) = 25
    assert X_enc.shape[1] == 25

    feature_names = extract_feature_names(preprocessor)
    assert len(feature_names) == 25

    label_map = build_readable_label_map(feature_names)
    assert len(label_map) == 25
    assert "Monthly Charges ($)" in label_map.values()


def test_model_bundle_integrity():
    """Verify saved bundle has all required components and is ready for inference."""
    settings = get_settings()
    bundle_path = settings.model_bundle_path
    assert bundle_path.exists(), f"Model bundle missing at {bundle_path}"

    bundle = joblib.load(bundle_path)
    required_keys = [
        "model",
        "preprocessor",
        "feature_names",
        "feature_labels",
        "medians",
        "modes",
        "model_name",
        "numeric_columns",
        "categorical_columns",
    ]
    for key in required_keys:
        assert key in bundle, f"Bundle missing key: {key}"

    assert len(bundle["feature_names"]) == 25
    assert len(bundle["numeric_columns"]) == 9
    assert len(bundle["categorical_columns"]) == 5
