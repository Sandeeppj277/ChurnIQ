"""
ChurnIQ — AI Recommendation & Messaging Orchestrator

Orchestrates retention recommendation and messaging workflows:
1. Attempts Gemini generation via Google GenAI.
2. Validates JSON payload structure.
3. Automatically falls back to deterministic rule-based generation on any error, timeout, or missing key.
4. Enforces Rules 4 & 5: AI failure never breaks the app, and every output is labeled.
"""

from typing import Any
import logging

from ai.client import GeminiClient
from ai.prompts.recommendation import (
    RECOMMENDATION_SYSTEM_INSTRUCTION,
    build_recommendation_prompt,
)
from ai.prompts.message import (
    MESSAGE_SYSTEM_INSTRUCTION,
    build_message_prompt,
)
from ai.recommendation.rule_based import (
    generate_rule_based_recommendation,
    generate_rule_based_message,
)

logger = logging.getLogger("churniq.ai.engine")


class RecommendationEngine:
    """Orchestrator for retention recommendations and customer messaging."""

    def __init__(self) -> None:
        self.gemini_client = GeminiClient()

    def generate_recommendation(
        self,
        customer: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate retention recommendation using Gemini first, fallback on error."""
        cid = customer.get("customer_id", "")

        if self.gemini_client.is_configured:
            prompt = build_recommendation_prompt(customer, prediction)
            result = self.gemini_client.generate_json(
                prompt=prompt,
                system_instruction=RECOMMENDATION_SYSTEM_INSTRUCTION,
            )

            # Validate that returned JSON contains all required fields
            if result and all(
                k in result for k in ["explanation", "recommended_action", "reason", "priority"]
            ):
                priority_val = str(result["priority"]).capitalize()
                if priority_val not in ["High", "Medium", "Low"]:
                    priority_val = "Medium"

                return {
                    "customer_id": cid,
                    "explanation": str(result["explanation"]),
                    "recommended_action": str(result["recommended_action"]),
                    "reason": str(result["reason"]),
                    "priority": priority_val,
                    "source": "ai_generated",
                    "error": None,
                }
            logger.info("Gemini output invalid or missing fields for customer %s, falling back.", cid)

        # Fallback to deterministic rule-based generation
        fallback = generate_rule_based_recommendation(customer, prediction)
        fallback["error"] = "Rule-based fallback active (Gemini API unavailable, rate-limited, or unconfigured)"
        return fallback

    def generate_message(
        self,
        customer: dict[str, Any],
        prediction: dict[str, Any],
        tone: str = "Professional",
    ) -> dict[str, Any]:
        """Generate outreach message using Gemini first, fallback on error."""
        cid = customer.get("customer_id", "")

        if self.gemini_client.is_configured:
            prompt = build_message_prompt(customer, prediction, tone)
            result = self.gemini_client.generate_json(
                prompt=prompt,
                system_instruction=MESSAGE_SYSTEM_INSTRUCTION,
            )

            if result and "message" in result and result["message"].strip():
                return {
                    "customer_id": cid,
                    "message": str(result["message"]).strip(),
                    "tone": tone,
                    "source": "ai_generated",
                    "error": None,
                }
            logger.info("Gemini message output invalid for customer %s, falling back.", cid)

        # Fallback to deterministic rule-based message
        fallback = generate_rule_based_message(customer, prediction, tone)
        fallback["error"] = "Rule-based fallback active (Gemini API unavailable or unconfigured)"
        return fallback
