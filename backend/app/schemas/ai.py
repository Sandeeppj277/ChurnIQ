"""
ChurnIQ — AI Endpoint Schemas

Defines request and response models for:
- POST /ai/recommendation
- POST /ai/message
- POST /ai/chat
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Requests ──────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """Request payload for POST /ai/recommendation."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=50, description="Customer ID to evaluate")


class MessageRequest(BaseModel):
    """Request payload for POST /ai/message."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=50, description="Customer ID to contact")
    tone: Literal["Professional", "Friendly", "Empathetic", "Short"] = Field(
        default="Professional", description="Desired tone of outreach message"
    )


class ChatRequest(BaseModel):
    """Request payload for POST /ai/chat."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=50, description="Customer ID context")
    question: str = Field(
        ..., min_length=1, max_length=1000, description="Question about this customer's churn data"
    )


# ── Responses ─────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    """Response payload for POST /ai/recommendation."""

    model_config = ConfigDict(extra="ignore")

    customer_id: str
    explanation: str = Field(..., description="Executive summary of why the customer may churn")
    recommended_action: str = Field(..., description="Actionable retention tactic")
    reason: str = Field(..., description="Business rationale supporting the action")
    priority: Literal["High", "Medium", "Low"] = Field(..., description="Urgency priority")
    source: Literal["ai_generated", "rule_based"] = Field(..., description="Origin of the output")
    error: Optional[str] = Field(default=None, description="Notice if rule-based fallback was triggered")


class MessageResponse(BaseModel):
    """Response payload for POST /ai/message."""

    model_config = ConfigDict(extra="ignore")

    customer_id: str
    message: str = Field(..., description="Drafted customer outreach message")
    tone: Literal["Professional", "Friendly", "Empathetic", "Short"] = Field(...)
    source: Literal["ai_generated", "rule_based"] = Field(...)
    error: Optional[str] = Field(default=None, description="Notice if rule-based fallback was triggered")


class ChatResponse(BaseModel):
    """Response payload for POST /ai/chat."""

    model_config = ConfigDict(extra="ignore")

    customer_id: str
    answer: str = Field(..., description="Grounded response about the customer")
    source: Literal["ai_generated", "rule_based"] = Field(...)
    error: Optional[str] = Field(default=None, description="Notice if rule-based fallback was triggered")
