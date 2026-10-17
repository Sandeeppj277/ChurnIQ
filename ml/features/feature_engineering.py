"""
ChurnIQ — Feature Engineering & Encoding

Constructs a ColumnTransformer that scales the 9 numeric features with StandardScaler
and encodes the 5 categorical features with OneHotEncoder(handle_unknown='ignore').
Yields exactly 25 model features with mapped human-readable labels for SHAP and UI.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.preprocessing.preprocess import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Build the 25-column ColumnTransformer according to Section 6.3."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


def extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract clean feature names from a fitted ColumnTransformer."""
    feature_names: list[str] = []

    # Numeric columns come first
    feature_names.extend(NUMERIC_COLUMNS)

    # Categorical columns via one-hot encoder
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    feature_names.extend(list(cat_features))

    return feature_names


def build_readable_label_map(feature_names: list[str]) -> dict[str, str]:
    """Map encoded feature names to human-readable labels for SHAP & UI display."""
    human_labels: dict[str, str] = {
        "age": "Age",
        "tenure": "Tenure (Months)",
        "monthly_charges": "Monthly Charges ($)",
        "total_charges": "Total Charges ($)",
        "usage_frequency": "Usage Frequency (sessions/wk)",
        "service_usage": "Service Usage (%)",
        "complaint_count": "Complaint Count",
        "support_calls": "Support Calls",
        "engagement_score": "Engagement Score",
    }

    prefix_map = {
        "gender_": "Gender: ",
        "subscription_type_": "Subscription: ",
        "contract_type_": "Contract: ",
        "payment_method_": "Payment Method: ",
        "payment_status_": "Payment Status: ",
    }

    for name in feature_names:
        if name in human_labels:
            continue
        mapped = False
        for prefix, label_prefix in prefix_map.items():
            if name.startswith(prefix):
                val = name[len(prefix):]
                human_labels[name] = f"{label_prefix}{val}"
                mapped = True
                break
        if not mapped:
            human_labels[name] = name.replace("_", " ").title()

    return human_labels
