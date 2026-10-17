"""
ChurnIQ — SQLAlchemy ORM Table Definitions

Four core tables:
1. Customer: Customer demographic, account, and behavioral attributes (upserted).
2. Prediction: Model predictions, probabilities, risk levels, and SHAP top factors.
3. Recommendation: AI-written or rule-based retention explanations and actions.
4. Message: Tone-tailored customer outreach messages.
"""

from datetime import datetime, timezone
import json
from typing import Any
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from backend.app.models.database import Base


def utcnow():
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class Customer(Base):
    """Customer profile table, upserted on each prediction."""

    __tablename__ = "customers"

    customer_id = Column(String(50), primary_key=True, index=True)
    customer_name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=False)
    age = Column(Integer, nullable=False)
    tenure = Column(Integer, nullable=False)
    subscription_type = Column(String(20), nullable=False)
    contract_type = Column(String(20), nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    payment_method = Column(String(30), nullable=False)
    payment_status = Column(String(20), nullable=False)
    usage_frequency = Column(Float, nullable=False)
    service_usage = Column(Float, nullable=False)
    complaint_count = Column(Integer, nullable=False)
    support_calls = Column(Integer, nullable=False)
    engagement_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utcnow, server_default=func.now())
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, server_default=func.now())

    # Relationships
    predictions = relationship("Prediction", back_populates="customer", cascade="all, delete-orphan", order_by="desc(Prediction.created_at)")
    recommendations = relationship("Recommendation", back_populates="customer", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="customer", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """Convert customer model to dictionary."""
        return {
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "gender": self.gender,
            "age": self.age,
            "tenure": self.tenure,
            "subscription_type": self.subscription_type,
            "contract_type": self.contract_type,
            "monthly_charges": self.monthly_charges,
            "total_charges": self.total_charges,
            "payment_method": self.payment_method,
            "payment_status": self.payment_status,
            "usage_frequency": self.usage_frequency,
            "service_usage": self.service_usage,
            "complaint_count": self.complaint_count,
            "support_calls": self.support_calls,
            "engagement_score": self.engagement_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Prediction(Base):
    """Model predictions log (every prediction adds a row)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    prediction = Column(String(30), nullable=False)  # "Likely to Churn" or "Not Likely to Churn"
    churn_probability = Column(Float, nullable=False)  # Percent, e.g. 78.45
    risk_level = Column(String(10), nullable=False, index=True)  # "LOW", "MEDIUM", "HIGH"
    _top_factors = Column("top_factors", Text, nullable=False)  # JSON-serialized list of 5 factors
    model_name = Column(String(50), nullable=False, default="XGBoost")
    created_at = Column(DateTime, default=utcnow, server_default=func.now(), index=True)

    # Relationships
    customer = relationship("Customer", back_populates="predictions")
    recommendations = relationship("Recommendation", back_populates="prediction", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="prediction", cascade="all, delete-orphan")

    @property
    def top_factors(self) -> list[dict[str, Any]]:
        """Deserialize JSON top factors."""
        if not self._top_factors:
            return []
        try:
            return json.loads(self._top_factors)
        except Exception:
            return []

    @top_factors.setter
    def top_factors(self, value: list[dict[str, Any]]) -> None:
        """Serialize top factors to JSON string."""
        self._top_factors = json.dumps(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "prediction": self.prediction,
            "churn_probability": round(float(self.churn_probability), 2),
            "risk_level": self.risk_level,
            "top_factors": self.top_factors,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Recommendation(Base):
    """Retention recommendations table (supports Gemini and rule-based)."""

    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    explanation = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    priority = Column(String(10), nullable=False)  # "High", "Medium", "Low"
    source = Column(String(20), nullable=False)  # "ai_generated" or "rule_based"
    created_at = Column(DateTime, default=utcnow, server_default=func.now())

    # Relationships
    customer = relationship("Customer", back_populates="recommendations")
    prediction = relationship("Prediction", back_populates="recommendations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "prediction_id": self.prediction_id,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
            "priority": self.priority,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Message(Base):
    """Personalized outreach messages (supports Gemini and rule-based)."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    tone = Column(String(20), nullable=False)  # "Professional", "Friendly", "Empathetic", "Short"
    source = Column(String(20), nullable=False)  # "ai_generated" or "rule_based"
    created_at = Column(DateTime, default=utcnow, server_default=func.now())

    # Relationships
    customer = relationship("Customer", back_populates="messages")
    prediction = relationship("Prediction", back_populates="messages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "prediction_id": self.prediction_id,
            "message": self.message,
            "tone": self.tone,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
