"""
ChurnIQ — Machine Learning Service

Singleton ML service that loads and caches:
- The trained model bundle (XGBoost / RandomForest)
- Preprocessor (ColumnTransformer)
- SHAP TreeExplainer

Enforces:
- Rule 1: Only the model predicts churn probability.
- Rule 2: Single clean_data() function called with saved training medians/modes.
- Rule 6: HTTP 503 raised if model bundle is not found.
- Rule 7: Model and SHAP explainer loaded and cached once at first use.
- Rule 8: Thresholds read from settings (LOW_RISK_MAX=39, MEDIUM_RISK_MAX=69).
"""

from pathlib import Path
from typing import Any
from fastapi import HTTPException, status
import joblib
import pandas as pd

from backend.app.config import get_settings
from ml.preprocessing.preprocess import clean_data
from ml.features.explainability import SHAPExplainerService


class MLService:
    """Singleton service for customer churn inference and SHAP explainability."""

    _instance: "MLService | None" = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.bundle: dict[str, Any] | None = None
        self.explainer_service: SHAPExplainerService | None = None
        self.model_ready: bool = False
        self._load_bundle_if_available()

    @classmethod
    def get_instance(cls) -> "MLService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_bundle_if_available(self) -> None:
        """Attempt to load model bundle and initialize SHAP explainer."""
        bundle_path: Path = self.settings.model_bundle_path
        if not bundle_path.exists():
            self.bundle = None
            self.explainer_service = None
            self.model_ready = False
            return

        try:
            self.bundle = joblib.load(bundle_path)
            model = self.bundle["model"]
            feature_names = self.bundle["feature_names"]
            feature_labels = self.bundle["feature_labels"]
            self.explainer_service = SHAPExplainerService(
                model=model,
                feature_names=feature_names,
                label_map=feature_labels,
            )
            self.model_ready = True
        except Exception as exc:
            self.model_ready = False
            self.bundle = None
            self.explainer_service = None
            print(f"[MLService] Error loading model bundle: {exc}")

    def reload(self) -> bool:
        """Force reload of the model bundle."""
        self._load_bundle_if_available()
        return self.model_ready

    def predict_customer(self, customer_data: dict[str, Any]) -> dict[str, Any]:
        """Compute churn probability, risk level, and top 5 SHAP factors.

        Raises:
            HTTPException 503 if model is not trained/loaded.
        """
        if not self.model_ready or self.bundle is None or self.explainer_service is None:
            # Check if model file was recently generated
            self._load_bundle_if_available()
            if not self.model_ready or self.bundle is None or self.explainer_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not trained. Run: python setup_project.py",
                )

        # 1. Convert input to a single-row DataFrame
        df_single = pd.DataFrame([customer_data])

        # 2. Clean data using identical clean_data() with training medians and modes
        df_cleaned, _, _ = clean_data(
            df_single,
            medians=self.bundle["medians"],
            modes=self.bundle["modes"],
            is_training=False,
        )

        # 3. Transform features via cached ColumnTransformer
        feature_cols = self.bundle["numeric_columns"] + self.bundle["categorical_columns"]
        X_encoded = self.bundle["preprocessor"].transform(df_cleaned[feature_cols])

        # 4. Predict churn probability using the classifier
        model = self.bundle["model"]
        prob_churn = float(model.predict_proba(X_encoded)[0, 1])
        churn_pct = round(prob_churn * 100.0, 2)

        # 5. Determine Risk Level using configured thresholds
        # LOW if <= LOW_RISK_MAX (39), MEDIUM if <= MEDIUM_RISK_MAX (69), else HIGH
        if churn_pct <= self.settings.LOW_RISK_MAX:
            risk_level = "LOW"
        elif churn_pct <= self.settings.MEDIUM_RISK_MAX:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # 6. Binary prediction label at 50% cutoff
        prediction_label = "Likely to Churn" if prob_churn >= 0.5 else "Not Likely to Churn"

        # 7. Compute top 5 SHAP factors
        top_factors = self.explainer_service.explain_instance(X_encoded, top_k=5)

        return {
            "customer_id": str(customer_data["customer_id"]),
            "prediction": prediction_label,
            "churn_probability": churn_pct,
            "risk_level": risk_level,
            "top_factors": top_factors,
            "model": self.bundle.get("model_name", "XGBoost"),
        }
