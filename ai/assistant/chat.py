"""
ChurnIQ — Grounded Customer Chat Assistant

Handles conversational Q&A regarding a specific customer's churn data:
- Tries Gemini with grounded prompt first.
- Falls back to rule-based contextual answer on error or missing API key.
"""

from typing import Any
import logging

from ai.client import GeminiClient
from ai.prompts.chat import (
    CHAT_SYSTEM_INSTRUCTION,
    build_chat_prompt,
)
from ai.recommendation.rule_based import generate_rule_based_chat_answer

logger = logging.getLogger("churniq.ai.chat")


class CustomerAssistant:
    """Conversational assistant strictly grounded in stored customer data."""

    def __init__(self) -> None:
        self.gemini_client = GeminiClient()

    def answer_question(
        self,
        customer: dict[str, Any],
        prediction: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """Answer question about customer churn data."""
        cid = customer.get("customer_id", "")

        if self.gemini_client.is_configured:
            prompt = build_chat_prompt(customer, prediction, question)
            answer_text = self.gemini_client.generate_text(
                prompt=prompt,
                system_instruction=CHAT_SYSTEM_INSTRUCTION,
            )

            if answer_text and answer_text.strip():
                return {
                    "customer_id": cid,
                    "answer": answer_text.strip(),
                    "source": "ai_generated",
                    "error": None,
                }
            logger.info("Gemini chat returned empty response for customer %s, falling back.", cid)

        # Fallback to rule-based grounded response
        fallback_answer = generate_rule_based_chat_answer(customer, prediction, question)
        return {
            "customer_id": cid,
            "answer": fallback_answer,
            "source": "rule_based",
            "error": "Rule-based fallback active (Gemini API unavailable or unconfigured)",
        }
