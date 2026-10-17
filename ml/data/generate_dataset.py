"""
ChurnIQ — Synthetic Customer Dataset Generator

Generates 5,000 synthetic customers with 17 features, then injects 15 duplicate rows,
invalid ages, and missing total_charges to yield a 5,015-row raw dataset.
Calibrated with seed 42 to produce ~23% churn rate.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_churn_dataset(output_path: Path | str | None = None) -> pd.DataFrame:
    """Generate 5,015 rows of synthetic customer churn data (seed=42)."""
    np.random.seed(42)
    n_samples = 5000

    # 1. Customer Identifiers
    customer_ids = [f"C{90000 + i}" for i in range(n_samples)]
    first_names = [
        "Aarav", "Aditi", "Alex", "Amara", "Carlos", "Chen", "Chloe", "David",
        "Elena", "Fatima", "George", "Grace", "Hassan", "Ishaan", "Jack",
        "Kavya", "Liam", "Maya", "Noah", "Olivia", "Priya", "Rahul", "Sarah",
        "Sophia", "Tariq", "Vikram", "Yuki", "Zainab"
    ]
    last_names = [
        "Sharma", "Patel", "Smith", "Johnson", "Williams", "Brown", "Jones",
        "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
        "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
        "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris"
    ]
    customer_names = [
        f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        for _ in range(n_samples)
    ]

    # 2. Demographics & Account Attributes
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.51, 0.49])
    ages = np.random.randint(18, 76, size=n_samples)
    tenures = np.random.exponential(scale=18, size=n_samples).astype(int)
    tenures = np.clip(tenures, 0, 72)

    subscription_types = np.random.choice(
        ["Basic", "Standard", "Premium"], size=n_samples, p=[0.45, 0.35, 0.20]
    )
    contract_types = np.random.choice(
        ["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )

    # 3. Financial Attributes
    base_charges = {
        "Basic": np.random.uniform(20.0, 50.0, size=n_samples),
        "Standard": np.random.uniform(50.0, 85.0, size=n_samples),
        "Premium": np.random.uniform(85.0, 120.0, size=n_samples),
    }
    monthly_charges = np.zeros(n_samples)
    for i, sub in enumerate(subscription_types):
        monthly_charges[i] = round(float(base_charges[sub][i]), 2)

    # total_charges = monthly_charges * max(tenure, 1) * Uniform(0.9, 1.0)
    variance_factor = np.random.uniform(0.9, 1.0, size=n_samples)
    total_charges = np.round(
        monthly_charges * np.maximum(tenures, 1) * variance_factor, 2
    )

    payment_methods = np.random.choice(
        ["Credit Card", "UPI", "Wallet", "Debit Card", "Net Banking"],
        size=n_samples,
        p=[0.30, 0.25, 0.15, 0.20, 0.10],
    )
    payment_statuses = np.random.choice(
        ["On Time", "Delayed", "Failed"], size=n_samples, p=[0.75, 0.18, 0.07]
    )

    # 4. Usage & Engagement Attributes
    usage_frequency = np.round(
        np.clip(np.random.normal(loc=14.0, scale=6.0, size=n_samples), 0.0, 45.0), 1
    )
    service_usage = np.round(
        np.clip(np.random.normal(loc=55.0, scale=20.0, size=n_samples), 0.0, 100.0), 1
    )
    complaint_counts = np.random.poisson(lam=0.7, size=n_samples)
    support_calls = np.random.poisson(lam=1.4, size=n_samples)
    engagement_scores = np.round(
        np.clip(np.random.normal(loc=58.0, scale=18.0, size=n_samples), 0.0, 100.0), 1
    )

    # 5. Churn Formula Calibration (~23% churn)
    # Drivers:
    # Contract: Month-to-Month (+), Two-Year (--)
    # Payment status: Failed (++), Delayed (+), On Time (-)
    # Complaints: (+)
    # Support calls: (+)
    # Monthly charges: (+)
    # Tenure: (-)
    # Usage / Engagement: (-)
    contract_risk = np.array([
        1.15 if c == "Month-to-Month" else (-0.5 if c == "One-Year" else -1.25)
        for c in contract_types
    ])
    payment_risk = np.array([
        1.35 if p == "Failed" else (0.65 if p == "Delayed" else -0.55)
        for p in payment_statuses
    ])
    sub_risk = np.array([
        0.15 if s == "Basic" else (0.0 if s == "Standard" else -0.15)
        for s in subscription_types
    ])

    z = (
        -1.35
        + contract_risk
        + payment_risk
        + sub_risk
        + 0.55 * complaint_counts
        + 0.28 * support_calls
        + 0.016 * (monthly_charges - 65.0)
        - 0.045 * tenures
        - 0.032 * (usage_frequency - 14.0)
        - 0.015 * (service_usage - 55.0)
        - 0.038 * (engagement_scores - 58.0)
        + np.random.normal(0, 0.45, size=n_samples)
    )

    probs = 1.0 / (1.0 + np.exp(-z))
    churn = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_name": customer_names,
        "gender": genders,
        "age": ages,
        "tenure": tenures,
        "subscription_type": subscription_types,
        "contract_type": contract_types,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "payment_method": payment_methods,
        "payment_status": payment_statuses,
        "usage_frequency": usage_frequency,
        "service_usage": service_usage,
        "complaint_count": complaint_counts,
        "support_calls": support_calls,
        "engagement_score": engagement_scores,
        "churn": churn,
    })

    # 6. Inject Anomaly & Noise Rows (as required by Section 6.1)
    # 15 duplicate rows
    dup_indices = np.random.choice(df.index, size=15, replace=False)
    duplicates = df.loc[dup_indices].copy()

    # Invalid ages in a few rows
    invalid_age_indices = np.random.choice(df.index, size=6, replace=False)
    df.loc[invalid_age_indices[:3], "age"] = -5
    df.loc[invalid_age_indices[3:], "age"] = 135

    # Missing total_charges in a few rows
    missing_charge_indices = np.random.choice(df.index, size=8, replace=False)
    df.loc[missing_charge_indices, "total_charges"] = np.nan

    # Append the 15 duplicates so raw file has exactly 5,015 rows
    df_raw = pd.concat([df, duplicates], ignore_index=True)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(out, index=False)

    return df_raw


if __name__ == "__main__":
    from backend.app.config import get_settings
    settings = get_settings()
    df_gen = generate_churn_dataset(settings.dataset_path)
    print(f"Generated {len(df_gen)} rows -> {settings.dataset_path}")
    print(f"Raw shape: {df_gen.shape}")
    print(f"Base churn rate (without duplicates): {df_gen['churn'].mean():.2%}")
