"""
ChurnIQ — Prediction Schemas

Defines response models for single-customer churn scoring and SHAP explainability factors.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class Factor(BaseModel):
    """Single influential feature driver computed via SHAP."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(..., description="Human-readable feature name")
    impact: float = Field(..., description="Absolute SHAP impact magnitude (log-odds units)")
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Directional effect on churn probability"
    )


class PredictionResponse(BaseModel):
    """Response returned by POST /predict."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., description="ID of the scored customer")
    prediction: Literal["Likely to Churn", "Not Likely to Churn"] = Field(
        ..., description="Binary classification label at 50% threshold"
    )
    churn_probability: float = Field(
        ..., description="Estimated churn probability expressed as a percentage (0.00 to 100.00)"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Assigned risk band based on configured thresholds"
    )
    top_factors: list[Factor] = Field(
        ..., min_length=5, max_length=5, description="Top 5 most influential SHAP drivers"
    )
    model: str = Field(default="XGBoost", description="Name of the model that generated prediction")
    prediction_id: Optional[int] = Field(
        default=None, description="Database ID of the persisted prediction row"
    )
