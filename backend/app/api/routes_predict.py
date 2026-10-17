"""
ChurnIQ — Prediction Route

POST /predict receives CustomerInput, validates fields with Pydantic,
runs inference and SHAP explainability via MLService, upserts the customer,
records the prediction in the database, and returns PredictionResponse.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.schemas.customer import CustomerInput
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.ml_service import MLService
from backend.app.services.persistence_service import PersistenceService

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_customer(
    customer_input: CustomerInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict churn probability for a single customer and log results."""
    ml_service = MLService.get_instance()
    cust_data = customer_input.model_dump()

    # 1. Compute prediction and SHAP explainability
    pred_result = ml_service.predict_customer(cust_data)

    # 2. Upsert customer in database
    PersistenceService.upsert_customer(db, cust_data)

    # 3. Save prediction record
    prediction_record = PersistenceService.save_prediction(
        db=db,
        customer_id=pred_result["customer_id"],
        prediction=pred_result["prediction"],
        churn_probability=pred_result["churn_probability"],
        risk_level=pred_result["risk_level"],
        top_factors=pred_result["top_factors"],
        model_name=pred_result["model"],
    )

    pred_result["prediction_id"] = prediction_record.id
    return PredictionResponse(**pred_result)
