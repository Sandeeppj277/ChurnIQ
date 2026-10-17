"""
ChurnIQ — Dashboard Summary Route

GET /dashboard/summary aggregates total predictions scored, risk distributions,
average churn probability, the 10 most recent predictions, and current model status.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.schemas.dashboard import DashboardSummary
from backend.app.services.persistence_service import PersistenceService

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    """Retrieve aggregated dashboard metrics."""
    summary_data = PersistenceService.get_dashboard_summary(db)
    return DashboardSummary(**summary_data)
