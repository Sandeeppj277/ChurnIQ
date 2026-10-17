"""
ChurnIQ — Dashboard Schemas

Defines schemas for GET /dashboard/summary metrics, risk distribution,
recent predictions, and the model snapshot.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RiskCounts(BaseModel):
    """Counts of predictions categorized into risk tiers."""

    model_config = ConfigDict(extra="ignore")

    LOW: int = Field(default=0, ge=0)
    MEDIUM: int = Field(default=0, ge=0)
    HIGH: int = Field(default=0, ge=0)


class RecentPredictionItem(BaseModel):
    """Summary item for a recently scored customer."""

    model_config = ConfigDict(extra="ignore")

    id: int
    customer_id: str
    customer_name: str
    prediction: str
    churn_probability: float
    risk_level: str
    created_at: str


class ModelSnapshot(BaseModel):
    """Current status and performance metrics of the production model."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_name: str = "XGBoost"
    accuracy: float = 0.0
    roc_auc: float = 0.0
    status: str = "ready"


class DashboardSummary(BaseModel):
    """Top-level payload returned by GET /dashboard/summary."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    total_scored: int = Field(..., ge=0, description="Total predictions recorded (counts prediction rows)")
    risk_counts: RiskCounts = Field(..., description="Distribution of risk levels across all predictions")
    average_probability: float = Field(..., ge=0.0, le=100.0, description="Mean churn probability across all scored rows")
    recent_predictions: list[RecentPredictionItem] = Field(
        default_factory=list, description="Latest 10 scored customers"
    )
    model_snapshot: ModelSnapshot = Field(..., description="Current model accuracy and status")
