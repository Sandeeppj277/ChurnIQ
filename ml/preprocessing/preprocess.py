"""
ChurnIQ — Preprocessing & Data Cleaning

Single cleaning function (clean_data) called identically by training and single-customer
prediction paths. Enforces Rule 2: "One cleaning function".
"""

from typing import Any
import pandas as pd
import numpy as np


NUMERIC_COLUMNS = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "usage_frequency",
    "service_usage",
    "complaint_count",
    "support_calls",
    "engagement_score",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "subscription_type",
    "contract_type",
    "payment_method",
    "payment_status",
]


def clean_data(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
    modes: dict[str, Any] | None = None,
    is_training: bool = False,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    """Clean customer data according to Section 6.2 of the specification.

    Execution order:
    1. Drop rows with duplicate customer_id (when customer_id is present).
    2. Compute/apply medians for numeric columns.
    3. Compute/apply modes for categorical columns.
    4. Replace ages outside [1, 110] with the median age.
    5. Set negative numeric values to 0.
    6. Clip engagement_score to [0, 100].

    Returns:
        (cleaned_df, medians, modes)
    """
    df_clean = df.copy()

    # 1. Drop duplicate customer_id if present
    if "customer_id" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["customer_id"], keep="first")

    # 2 & 3. Compute medians/modes if training, or use supplied values
    computed_medians: dict[str, float] = {} if medians is None else dict(medians)
    computed_modes: dict[str, Any] = {} if modes is None else dict(modes)

    if is_training:
        for col in NUMERIC_COLUMNS:
            if col in df_clean.columns:
                valid_vals = df_clean[col].dropna()
                if col == "age":
                    # Filter valid ages for median computation
                    valid_ages = valid_vals[(valid_vals >= 1) & (valid_vals <= 110)]
                    computed_medians[col] = float(valid_ages.median()) if not valid_ages.empty else 40.0
                else:
                    computed_medians[col] = float(valid_vals.median()) if not valid_vals.empty else 0.0

        for col in CATEGORICAL_COLUMNS:
            if col in df_clean.columns:
                mode_series = df_clean[col].mode(dropna=True)
                computed_modes[col] = mode_series.iloc[0] if not mode_series.empty else "Standard"

    # Fill missing numeric values with median
    for col in NUMERIC_COLUMNS:
        if col in df_clean.columns:
            fill_val = computed_medians.get(col, 0.0)
            df_clean[col] = df_clean[col].fillna(fill_val)

    # Fill missing categorical values with mode
    for col in CATEGORICAL_COLUMNS:
        if col in df_clean.columns:
            fill_val = computed_modes.get(col, "Unknown")
            df_clean[col] = df_clean[col].fillna(fill_val)

    # 4. Replace ages outside [1, 110] with median age
    if "age" in df_clean.columns:
        median_age = computed_medians.get("age", 40.0)
        invalid_mask = (df_clean["age"] < 1) | (df_clean["age"] > 110)
        df_clean.loc[invalid_mask, "age"] = median_age

    # 5. Set negative numeric values to 0
    for col in NUMERIC_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.0)

    # 6. Clip engagement_score to [0, 100]
    if "engagement_score" in df_clean.columns:
        df_clean["engagement_score"] = df_clean["engagement_score"].clip(lower=0.0, upper=100.0)

    # Also ensure service_usage is within [0, 100]
    if "service_usage" in df_clean.columns:
        df_clean["service_usage"] = df_clean["service_usage"].clip(lower=0.0, upper=100.0)

    return df_clean, computed_medians, computed_modes
