"""
Application configuration.

Loads settings from environment variables with sensible defaults.
No hardcoded API keys or connection strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from config/)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""

    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    )
    openai_temperature: float = 0.0
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            "postgresql://deepmed:deepmed@localhost:5432/deepmed?sslmode=disable",
        )
    )

    def validate(self) -> None:
        """Raise if critical settings are missing."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it before starting the application."
            )
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set it before starting the application."
            )


def get_settings() -> Settings:
    """Factory that returns a validated Settings instance."""
    settings = Settings()
    settings.validate()
    return settings
