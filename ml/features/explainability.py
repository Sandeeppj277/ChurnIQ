"""
ChurnIQ — Explainability & SHAP Service

Instantiates and caches shap.TreeExplainer.
Extracts top 5 directional drivers per single-customer prediction, and calculates
global feature importance for model evaluation.
"""

from typing import Any
import numpy as np
import shap


class SHAPExplainerService:
    """Wrapper around shap.TreeExplainer with singleton caching."""

    def __init__(self, model: Any, feature_names: list[str], label_map: dict[str, str]):
        self.model = model
        self.feature_names = feature_names
        self.label_map = label_map
        # Build TreeExplainer once
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, x_transformed: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Compute the top K SHAP drivers for a single customer.

        Args:
            x_transformed: 2D array of shape (1, 25) with scaled & encoded features.
            top_k: Number of top factors to return (default 5).

        Returns:
            List of dicts with:
                feature: Human-readable feature label
                impact: Absolute SHAP magnitude (log-odds impact, rounded to 4 decimals)
                direction: 'increases_risk' if positive, 'decreases_risk' if negative
        """
        # Ensure 2D
        if x_transformed.ndim == 1:
            x_transformed = x_transformed.reshape(1, -1)

        raw_shap = self.explainer.shap_values(x_transformed)

        # Handle different SHAP output shapes across versions
        if isinstance(raw_shap, list):
            # Binary classification list [class 0, class 1]
            values = np.array(raw_shap[1]).flatten()
        elif hasattr(raw_shap, "values"):
            # Explanation object
            values = np.array(raw_shap.values).flatten()
        else:
            values = np.array(raw_shap).flatten()

        # Build list of feature impacts
        factors = []
        for i, val in enumerate(values):
            if i >= len(self.feature_names):
                break
            feat_name = self.feature_names[i]
            label = self.label_map.get(feat_name, feat_name)
            float_val = float(val)
            factors.append({
                "feature": label,
                "impact": round(abs(float_val), 4),
                "direction": "increases_risk" if float_val > 0 else "decreases_risk",
                "_raw_val": float_val,
            })

        # Sort descending by impact (absolute magnitude)
        factors.sort(key=lambda item: item["impact"], reverse=True)

        # Return top K, stripping internal helper keys
        top_factors = [
            {
                "feature": f["feature"],
                "impact": f["impact"],
                "direction": f["direction"],
            }
            for f in factors[:top_k]
        ]
        return top_factors


def compute_global_importance(
    model: Any,
    feature_names: list[str],
    label_map: dict[str, str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Compute normalized global feature importance (top 10)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Fallback if no feature_importances_ attribute
        importances = np.ones(len(feature_names)) / len(feature_names)

    ranked_indices = np.argsort(importances)[::-1][:top_n]
    result = []
    for idx in ranked_indices:
        feat_name = feature_names[idx]
        result.append({
            "feature": label_map.get(feat_name, feat_name),
            "importance": round(float(importances[idx]), 4),
        })
    return result
