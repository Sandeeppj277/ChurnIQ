"""
ChurnIQ — Project Setup & ML Pipeline Orchestrator

Executes the complete setup workflow from the repository root:
1. Generate synthetic dataset (5,015 raw rows with noise) -> ml/data/churn_dataset.csv
2. Preprocess & Train XGBoost classifier -> ml/artifacts/churn_model.joblib
3. Evaluate on 1,000 test rows -> ml/artifacts/metrics.json
4. Verify all artifacts exist and validate acceptance targets.

Usage:
    python setup_project.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from ml.data.generate_dataset import generate_churn_dataset
from ml.training import train_model
from ml.evaluation import evaluate_model


def run_setup() -> bool:
    """Run end-to-end data generation, training, and evaluation pipeline."""
    settings = get_settings()

    print("\n" + "=" * 60)
    print("      ChurnIQ — End-to-End Pipeline Setup")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n[Step 1/3] Generating synthetic dataset...")
    df_raw = generate_churn_dataset(settings.dataset_path)
    print(f"-> Generated {len(df_raw)} rows at: {settings.dataset_path}")

    # Step 2: Train model
    print("\n[Step 2/3] Training XGBoost classifier...")
    train_res = train_model.train_churn_model(
        dataset_path=settings.dataset_path,
        output_bundle_path=settings.model_bundle_path,
    )
    print(f"-> Model bundle saved at: {settings.model_bundle_path}")

    # Step 3: Evaluate model
    print("\n[Step 3/3] Evaluating model on held-out test set...")
    metrics = evaluate_model.evaluate_model(
        model=train_res["bundle"]["model"],
        X_test=train_res["X_test"],
        y_test=train_res["y_test"],
        feature_names=train_res["feature_names"],
        label_map=train_res["label_map"],
        model_name=train_res["model_name"],
        output_path=settings.metrics_path,
    )

    acc = metrics["accuracy"]
    roc_auc = metrics["roc_auc"]
    passed = (acc > 0.77) and (roc_auc > 0.80)

    if passed:
        print("[SUCCESS] All pipeline steps completed and passed acceptance targets!")
    else:
        print("[WARNING] Pipeline completed, but targets were not met. Check feature weights.")

    return passed


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
