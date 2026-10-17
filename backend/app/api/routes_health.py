"""
ChurnIQ — Health Check Route

GET /health returns:
- status ("ok")
- model_ready (bool)
- ai_enabled (bool)
"""

from typing import Any
from fastapi import APIRouter
from backend.app.config import get_settings
from backend.app.services.ml_service import MLService

router = APIRouter(tags=["Health"])


@router.get("/health")
def get_health() -> dict[str, Any]:
    """Health check verifying API operational status, model availability, and AI setup."""
    settings = get_settings()
    ml_service = MLService.get_instance()

    return {
        "status": "ok",
        "model_ready": ml_service.model_ready,
        "ai_enabled": settings.ai_enabled,
    }
