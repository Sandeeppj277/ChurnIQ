"""
ChurnIQ — Application Configuration

All settings are read from environment variables (loaded via .env).
Pydantic BaseSettings ensures type safety and provides defaults.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Resolve project root (two levels up from this file) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Central configuration sourced from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Gemini AI ────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    AI_MODEL: str = "gemini-3.8-flash"

    # ── Risk Thresholds (percent) ────────────────────────────────
    LOW_RISK_MAX: int = 39
    MEDIUM_RISK_MAX: int = 69

    # ── Database ─────────────────────────────────────────────────
    # Default: SQLite file in project root.
    # Override with a PostgreSQL connection string to switch engines.
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT / 'churn_dev.db'}"

    # ── Frontend CORS ────────────────────────────────────────────
    FRONTEND_URL: str = "http://127.0.0.1:5500"

    # ── Derived Paths (not from .env) ────────────────────────────
    @property
    def model_bundle_path(self) -> Path:
        return PROJECT_ROOT / "ml" / "artifacts" / "churn_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return PROJECT_ROOT / "ml" / "artifacts" / "metrics.json"

    @property
    def dataset_path(self) -> Path:
        return PROJECT_ROOT / "ml" / "data" / "churn_dataset.csv"

    @property
    def ai_enabled(self) -> bool:
        """True only when a Gemini API key is actually configured."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
