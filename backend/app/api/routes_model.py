"""
ChurnIQ — Model Metrics Route

GET /model/metrics serves the contents of ml/artifacts/metrics.json,
including test accuracy, precision, recall, F1-score, ROC-AUC,
confusion matrix, and top 10 global feature importances.
"""

import json
from typing import Any
from fastapi import APIRouter, HTTPException, status

from backend.app.config import get_settings

router = APIRouter(tags=["Model Performance"])


@router.get("/model/metrics")
def get_model_metrics() -> dict[str, Any]:
    """Retrieve test evaluation metrics and global feature importances."""
    settings = get_settings()

    if not settings.metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not trained. Run: python setup_project.py",
        )

    try:
        with open(settings.metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load model metrics",
        ) from exc
