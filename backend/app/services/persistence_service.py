"""
ChurnIQ — Persistence Service

Handles database interactions:
- Upserting customer profile records
- Logging every prediction
- Storing AI & rule-based recommendations and messages
- Generating dashboard summary aggregations
"""

from datetime import datetime, timezone
import json
from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.config import get_settings
from backend.app.models.tables import Customer, Prediction, Recommendation, Message


class PersistenceService:
    """Service handling all database persistence operations."""

    @staticmethod
    def upsert_customer(db: Session, customer_data: dict[str, Any]) -> Customer:
        """Upsert a customer record by customer_id.

        If the customer exists, updates attributes; otherwise creates a new record.
        """
        cid = str(customer_data["customer_id"]).strip()
        customer = db.query(Customer).filter(Customer.customer_id == cid).first()

        if customer:
            # Update existing customer
            customer.customer_name = customer_data.get("customer_name", customer.customer_name)
            customer.gender = customer_data.get("gender", customer.gender)
            customer.age = int(customer_data.get("age", customer.age))
            customer.tenure = int(customer_data.get("tenure", customer.tenure))
            customer.subscription_type = customer_data.get("subscription_type", customer.subscription_type)
            customer.contract_type = customer_data.get("contract_type", customer.contract_type)
            customer.monthly_charges = float(customer_data.get("monthly_charges", customer.monthly_charges))
            customer.total_charges = float(customer_data.get("total_charges", customer.total_charges))
            customer.payment_method = customer_data.get("payment_method", customer.payment_method)
            customer.payment_status = customer_data.get("payment_status", customer.payment_status)
            customer.usage_frequency = float(customer_data.get("usage_frequency", customer.usage_frequency))
            customer.service_usage = float(customer_data.get("service_usage", customer.service_usage))
            customer.complaint_count = int(customer_data.get("complaint_count", customer.complaint_count))
            customer.support_calls = int(customer_data.get("support_calls", customer.support_calls))
            customer.engagement_score = float(customer_data.get("engagement_score", customer.engagement_score))
            customer.updated_at = datetime.now(timezone.utc)
        else:
            # Create new customer record
            customer = Customer(
                customer_id=cid,
                customer_name=customer_data.get("customer_name", cid),
                gender=customer_data.get("gender", "Female"),
                age=int(customer_data.get("age", 40)),
                tenure=int(customer_data.get("tenure", 12)),
                subscription_type=customer_data.get("subscription_type", "Standard"),
                contract_type=customer_data.get("contract_type", "Month-to-Month"),
                monthly_charges=float(customer_data.get("monthly_charges", 50.0)),
                total_charges=float(customer_data.get("total_charges", 600.0)),
                payment_method=customer_data.get("payment_method", "Credit Card"),
                payment_status=customer_data.get("payment_status", "On Time"),
                usage_frequency=float(customer_data.get("usage_frequency", 10.0)),
                service_usage=float(customer_data.get("service_usage", 50.0)),
                complaint_count=int(customer_data.get("complaint_count", 0)),
                support_calls=int(customer_data.get("support_calls", 0)),
                engagement_score=float(customer_data.get("engagement_score", 50.0)),
            )
            db.add(customer)

        db.commit()
        db.refresh(customer)
        return customer

    @staticmethod
    def save_prediction(
        db: Session,
        customer_id: str,
        prediction: str,
        churn_probability: float,
        risk_level: str,
        top_factors: list[dict[str, Any]],
        model_name: str = "XGBoost",
    ) -> Prediction:
        """Log a new prediction row for a customer."""
        pred = Prediction(
            customer_id=customer_id,
            prediction=prediction,
            churn_probability=round(float(churn_probability), 2),
            risk_level=risk_level,
            model_name=model_name,
        )
        pred.top_factors = top_factors
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return pred

    @staticmethod
    def save_recommendation(
        db: Session,
        customer_id: str,
        prediction_id: int,
        explanation: str,
        recommended_action: str,
        reason: str,
        priority: str,
        source: str,
    ) -> Recommendation:
        """Persist a retention recommendation."""
        rec = Recommendation(
            customer_id=customer_id,
            prediction_id=prediction_id,
            explanation=explanation,
            recommended_action=recommended_action,
            reason=reason,
            priority=priority,
            source=source,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec

    @staticmethod
    def save_message(
        db: Session,
        customer_id: str,
        prediction_id: int,
        message_content: str,
        tone: str,
        source: str,
    ) -> Message:
        """Persist an outreach message."""
        msg = Message(
            customer_id=customer_id,
            prediction_id=prediction_id,
            message=message_content,
            tone=tone,
            source=source,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def get_customer(db: Session, customer_id: str) -> Customer | None:
        """Look up a customer by customer_id."""
        return db.query(Customer).filter(Customer.customer_id == customer_id).first()

    @staticmethod
    def get_latest_prediction(db: Session, customer_id: str) -> Prediction | None:
        """Fetch the most recent prediction for a customer."""
        return (
            db.query(Prediction)
            .filter(Prediction.customer_id == customer_id)
            .order_by(desc(Prediction.created_at))
            .first()
        )

    @staticmethod
    def get_dashboard_summary(db: Session) -> dict[str, Any]:
        """Aggregate summary metrics for the Dashboard view."""
        settings = get_settings()

        # 1. Total predictions count (Rule: counts prediction rows, not distinct customers)
        total_scored = db.query(func.count(Prediction.id)).scalar() or 0

        # 2. Counts by risk level
        risk_counts = {
            "LOW": db.query(func.count(Prediction.id)).filter(Prediction.risk_level == "LOW").scalar() or 0,
            "MEDIUM": db.query(func.count(Prediction.id)).filter(Prediction.risk_level == "MEDIUM").scalar() or 0,
            "HIGH": db.query(func.count(Prediction.id)).filter(Prediction.risk_level == "HIGH").scalar() or 0,
        }

        # 3. Average churn probability
        avg_prob = db.query(func.avg(Prediction.churn_probability)).scalar()
        average_probability = round(float(avg_prob), 2) if avg_prob is not None else 0.0

        # 4. Recent predictions (latest 10)
        recent_rows = (
            db.query(Prediction, Customer.customer_name)
            .join(Customer, Prediction.customer_id == Customer.customer_id)
            .order_by(desc(Prediction.created_at))
            .limit(10)
            .all()
        )

        recent_predictions = [
            {
                "id": p.id,
                "customer_id": p.customer_id,
                "customer_name": c_name,
                "prediction": p.prediction,
                "churn_probability": round(float(p.churn_probability), 2),
                "risk_level": p.risk_level,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else "",
            }
            for p, c_name in recent_rows
        ]

        # 5. Model snapshot from metrics.json if present
        model_snapshot = {
            "model_name": "XGBoost",
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "status": "ready" if settings.model_bundle_path.exists() else "not_trained",
        }
        if settings.metrics_path.exists():
            try:
                with open(settings.metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                    model_snapshot.update({
                        "model_name": metrics_data.get("model_name", "XGBoost"),
                        "accuracy": metrics_data.get("accuracy", 0.0),
                        "roc_auc": metrics_data.get("roc_auc", 0.0),
                    })
            except Exception:
                pass

        return {
            "total_scored": total_scored,
            "risk_counts": risk_counts,
            "average_probability": average_probability,
            "recent_predictions": recent_predictions,
            "model_snapshot": model_snapshot,
        }
