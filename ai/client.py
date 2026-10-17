"""
ChurnIQ — Gemini API Client Wrapper

Initializes Google GenAI client from settings (GEMINI_API_KEY, AI_MODEL).
Enforces:
- Rule 1: Models only explain/recommend and cannot modify predictions.
- Rule 3: Key stays on server from .env.
- Rule 4: Traps all network, timeout, API, and JSON parse errors gracefully.
"""

import json
import logging
import re
from typing import Any
from google import genai
from google.genai import types

from backend.app.config import get_settings

logger = logging.getLogger("churniq.ai.client")


class GeminiClient:
    """Wrapper around Google GenAI client with defensive error handling."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: genai.Client | None = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Google GenAI client if an API key is present."""
        key = self.settings.GEMINI_API_KEY
        if key and key.strip():
            try:
                self._client = genai.Client(api_key=key.strip())
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI client: %s", exc)
                self._client = None
        else:
            self._client = None

    @property
    def is_configured(self) -> bool:
        """True if client is initialized with an active key."""
        return self._client is not None

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate structured JSON response using Gemini.

        Returns None on any error, allowing the caller to trigger rule-based fallback.
        """
        if not self.is_configured:
            return None

        model_name = model or self.settings.AI_MODEL

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or "{}"
            # Clean possible markdown wrapping if returned
            clean_text = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
            clean_text = re.sub(r"^```\s*$", "", clean_text, flags=re.MULTILINE)
            return json.loads(clean_text)
        except Exception as exc:
            logger.warning("Gemini JSON generation failed: %s", exc)
            return None

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str | None = None,
    ) -> str | None:
        """Generate free-form text response using Gemini.

        Returns None on any error, allowing the caller to trigger rule-based fallback.
        """
        if not self.is_configured:
            return None

        model_name = model or self.settings.AI_MODEL

        config = types.GenerateContentConfig(temperature=0.4)
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text.strip() if response.text else None
        except Exception as exc:
            logger.warning("Gemini text generation failed: %s", exc)
            return None
