"""
ChurnIQ — Pydantic Schemas Package
"""

from backend.app.schemas.customer import CustomerInput
from backend.app.schemas.prediction import PredictionResponse, Factor
from backend.app.schemas.dashboard import (
    DashboardSummary,
    RiskCounts,
    RecentPredictionItem,
    ModelSnapshot,
)
from backend.app.schemas.ai import (
    RecommendationRequest,
    RecommendationResponse,
    MessageRequest,
    MessageResponse,
    ChatRequest,
    ChatResponse,
)

__all__ = [
    "CustomerInput",
    "PredictionResponse",
    "Factor",
    "DashboardSummary",
    "RiskCounts",
    "RecentPredictionItem",
    "ModelSnapshot",
    "RecommendationRequest",
    "RecommendationResponse",
    "MessageRequest",
    "MessageResponse",
    "ChatRequest",
    "ChatResponse",
]
