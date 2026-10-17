"""
ChurnIQ — Customer Input Schema

Pydantic schema for single-customer input data.
Validates types, value ranges, and enums upfront before any inference or DB write.
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    """Customer profile input for prediction and persistence."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Unique customer identifier (letters, digits, underscore, hyphen)",
        examples=["C90042"],
    )
    customer_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Full customer name",
        examples=["Sarah Jenkins"],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender",
        examples=["Female"],
    )
    age: int = Field(
        ...,
        ge=1,
        le=110,
        description="Customer age in years (1 to 110)",
        examples=[42],
    )
    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months customer has been with the service",
        examples=[18],
    )
    subscription_type: Literal["Basic", "Standard", "Premium"] = Field(
        ...,
        description="Tier of service plan",
        examples=["Standard"],
    )
    contract_type: Literal["Month-to-Month", "One-Year", "Two-Year"] = Field(
        ...,
        description="Contract duration term",
        examples=["Month-to-Month"],
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        description="Current monthly bill amount in USD",
        examples=[64.50],
    )
    total_charges: float = Field(
        ...,
        ge=0.0,
        description="Lifetime bill amount in USD",
        examples=[1161.00],
    )
    payment_method: Literal[
        "Credit Card", "UPI", "Wallet", "Debit Card", "Net Banking"
    ] = Field(
        ...,
        description="Preferred payment instrument",
        examples=["Credit Card"],
    )
    payment_status: Literal["On Time", "Delayed", "Failed"] = Field(
        ...,
        description="Status of most recent payment",
        examples=["On Time"],
    )
    usage_frequency: float = Field(
        ...,
        ge=0.0,
        description="Average active usage sessions per week",
        examples=[14.2],
    )
    service_usage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of plan allowance or bandwidth utilized (0-100)",
        examples=[58.0],
    )
    complaint_count: int = Field(
        ...,
        ge=0,
        description="Lifetime formal complaints submitted",
        examples=[1],
    )
    support_calls: int = Field(
        ...,
        ge=0,
        description="Total support tickets or calls logged",
        examples=[2],
    )
    engagement_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Calculated behavioral engagement score (0-100)",
        examples=[62.5],
    )
