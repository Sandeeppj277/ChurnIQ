"""
ChurnIQ — AI Routes

Exposes:
- POST /ai/recommendation (generates & persists retention strategy)
- POST /ai/message (generates & persists tone-adjusted customer outreach)
- POST /ai/chat (answers customer-specific questions grounded in DB data)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.services.persistence_service import PersistenceService
from backend.app.schemas.ai import (
    RecommendationRequest,
    RecommendationResponse,
    MessageRequest,
    MessageResponse,
    ChatRequest,
    ChatResponse,
)
from ai.recommendation.engine import RecommendationEngine
from ai.assistant.chat import CustomerAssistant

router = APIRouter(prefix="/ai", tags=["Generative AI"])

# Singleton engines
recommendation_engine = RecommendationEngine()
customer_assistant = CustomerAssistant()


def _get_customer_and_prediction(db: Session, customer_id: str):
    """Retrieve customer and latest prediction or raise 404."""
    customer = PersistenceService.get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found. Please predict this customer first.",
        )

    prediction = PersistenceService.get_latest_prediction(db, customer_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction found for customer '{customer_id}'. Please run prediction first.",
        )

    return customer, prediction


@router.post("/recommendation", response_model=RecommendationResponse)
def get_recommendation(
    payload: RecommendationRequest,
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    """Generate and persist personalized retention recommendations."""
    customer, prediction = _get_customer_and_prediction(db, payload.customer_id)

    # Generate recommendation (Gemini with rule-based fallback)
    rec_data = recommendation_engine.generate_recommendation(
        customer=customer.to_dict(),
        prediction=prediction.to_dict(),
    )

    # Persist recommendation to database
    PersistenceService.save_recommendation(
        db=db,
        customer_id=customer.customer_id,
        prediction_id=prediction.id,
        explanation=rec_data["explanation"],
        recommended_action=rec_data["recommended_action"],
        reason=rec_data["reason"],
        priority=rec_data["priority"],
        source=rec_data["source"],
    )

    return RecommendationResponse(**rec_data)


@router.post("/message", response_model=MessageResponse)
def get_message(
    payload: MessageRequest,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Generate and persist personalized customer outreach message."""
    customer, prediction = _get_customer_and_prediction(db, payload.customer_id)

    # Generate message (Gemini with rule-based fallback)
    msg_data = recommendation_engine.generate_message(
        customer=customer.to_dict(),
        prediction=prediction.to_dict(),
        tone=payload.tone,
    )

    # Persist message to database
    PersistenceService.save_message(
        db=db,
        customer_id=customer.customer_id,
        prediction_id=prediction.id,
        message_content=msg_data["message"],
        tone=msg_data["tone"],
        source=msg_data["source"],
    )

    return MessageResponse(**msg_data)


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Answer natural language question about customer churn data."""
    customer, prediction = _get_customer_and_prediction(db, payload.customer_id)

    # Generate grounded answer (Gemini with rule-based fallback)
    chat_data = customer_assistant.answer_question(
        customer=customer.to_dict(),
        prediction=prediction.to_dict(),
        question=payload.question,
    )

    return ChatResponse(**chat_data)
