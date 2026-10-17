"""
ChurnIQ — Model Evaluation

Evaluates trained model on the 1,000 held-out test rows at threshold 0.5.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix,
and Top 10 Global Feature Importances.
Saves metrics to ml/artifacts/metrics.json for the GET /model/metrics endpoint.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from backend.app.config import get_settings
from ml.features.explainability import compute_global_importance


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    label_map: dict[str, str],
    model_name: str = "XGBoost",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute and record test metrics."""
    settings = get_settings()
    out_file = Path(output_path or settings.metrics_path)

    # Predictions at threshold 0.5
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Core classification metrics
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs))

    # Confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Global feature importance (top 10)
    top_importance = compute_global_importance(model, feature_names, label_map, top_n=10)

    metrics_payload = {
        "model_name": model_name,
        "test_samples": int(len(y_test)),
        "threshold": 0.5,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "global_feature_importance": top_importance,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 50)
    print("      ChurnIQ Model Evaluation Summary")
    print("=" * 50)
    print(f"Model:              {model_name}")
    print(f"Test Samples:       {len(y_test)}")
    print(f"Accuracy:           {acc * 100:.2f}%  (Acceptance Target: > 77.0%)")
    print(f"Precision:          {prec * 100:.2f}%")
    print(f"Recall:             {rec * 100:.2f}%")
    print(f"F1-Score:           {f1 * 100:.2f}%")
    print(f"ROC-AUC:            {roc_auc:.4f}  (Acceptance Target: > 0.800)")
    print("-" * 50)
    print(f"Confusion Matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved metrics to:   {out_file}")

    # Acceptance check
    acc_pass = acc > 0.77
    auc_pass = roc_auc > 0.80
    if acc_pass and auc_pass:
        print(">>> ACCEPTANCE CRITERIA: PASSED <<<")
    else:
        print(">>> WARNING: Targets not met (Acc > 77%, AUC > 0.80) <<<")
    print("=" * 50 + "\n")

    return metrics_payload
