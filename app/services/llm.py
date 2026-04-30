"""
LLM service factory.

Single place to configure and obtain ChatOpenAI instances.
No global mutable state — uses a module-level factory with lazy init.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config.settings import Settings, get_settings


@lru_cache(maxsize=1)
def _get_settings_cached() -> Settings:
    """Cache settings so env is read once."""
    return get_settings()


def get_llm(
    *,
    temperature: float | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.

    Parameters
    ----------
    temperature : float, optional
        Override the default temperature (0.0).
    model : str, optional
        Override the default model from settings.
    """
    settings = _get_settings_cached()
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model or settings.openai_model,
        temperature=temperature if temperature is not None else settings.openai_temperature,
    )
