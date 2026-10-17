"""
ChurnIQ — Model Training Pipeline

Trains XGBoost classifier (with automatic Random Forest fallback) using stratified 80/20 split.
Saves complete joblib bundle with model, preprocessor, feature names, label map,
and training medians/modes to ml/artifacts/churn_model.joblib.
"""

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.config import get_settings
from ml.preprocessing.preprocess import clean_data, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from ml.features.feature_engineering import (
    build_preprocessor,
    extract_feature_names,
    build_readable_label_map,
)


def train_churn_model(
    dataset_path: Path | str | None = None,
    output_bundle_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute model training pipeline and save bundle.

    Returns:
        dict containing trained model, preprocessor, test split, and bundle metadata.
    """
    settings = get_settings()
    data_file = Path(dataset_path or settings.dataset_path)
    bundle_file = Path(output_bundle_path or settings.model_bundle_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}. Run dataset generator first.")

    print(f"[Training] Loading raw data from: {data_file}")
    df_raw = pd.read_csv(data_file)
    print(f"[Training] Raw rows loaded: {len(df_raw)}")

    # 1. Clean data using single cleaning function in training mode
    df_clean, medians, modes = clean_data(df_raw, is_training=True)
    print(f"[Training] Cleaned rows after duplicate removal: {len(df_clean)}")

    X = df_clean[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = df_clean["churn"]

    # 2. Stratified 80/20 split (4,000 train, 1,000 test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Training] Split: {len(X_train_raw)} train rows, {len(X_test_raw)} test rows")

    # 3. Fit preprocessor on training data only
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = extract_feature_names(preprocessor)
    label_map = build_readable_label_map(feature_names)
    print(f"[Training] Features encoded: {len(feature_names)} columns")

    # 4. Train classifier: XGBoost (with RandomForest fallback)
    model_name = "XGBoost"
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        print("[Training] Training XGBClassifier...")
        model.fit(X_train, y_train)
    except Exception as exc:
        print(f"[Training] Warning: XGBoost unavailable ({exc}), falling back to RandomForestClassifier")
        from sklearn.ensemble import RandomForestClassifier
        model_name = "Random Forest"
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # 5. Assemble bundle
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "feature_labels": label_map,
        "medians": medians,
        "modes": modes,
        "model_name": model_name,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }

    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_file)
    print(f"[Training] Saved model bundle to: {bundle_file}")

    return {
        "bundle": bundle,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "feature_names": feature_names,
        "label_map": label_map,
        "model_name": model_name,
    }


if __name__ == "__main__":
    train_churn_model()
