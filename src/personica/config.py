"""Centralized Personica settings, loaded once from the environment / .env.

`PERSONICA_*` variables are the primary names; legacy `ASSISTANT_*` names are
accepted as fallbacks for backwards compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str, legacy: str | None = None) -> str:
    for candidate in (name, legacy):
        if candidate:
            value = os.getenv(candidate, "").strip()
            if value:
                return value
    return default


def _env_int(name: str, default: int, legacy: str | None = None) -> int:
    try:
        return int(_env_str(name, "", legacy))
    except ValueError:
        return default


def _env_float(name: str, default: float, legacy: str | None = None) -> float:
    try:
        return float(_env_str(name, "", legacy))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Storage
    data_dir: str
    embed_model: str

    # LLM (OpenRouter via LiteLLM)
    api_key: str
    chat_model: str
    utility_model: str
    base_url: str

    # Memory behaviour
    keep_last_turns: int
    retrieval_top_k: int
    retrieval_min_score: float
    relevance_weight: float          # hybrid rank: weight on similarity
    recency_half_life_days: float    # hybrid rank: recency decay half-life

    # Misc
    timezone: str
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=_env_str(
                "PERSONICA_DATA_DIR", "./personica_data",
                legacy="ASSISTANT_DATA_DIR",
            ),
            embed_model=_env_str(
                "PERSONICA_EMBED_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
                legacy="ASSISTANT_EMBED_MODEL",
            ),
            api_key=_env_str("OPENROUTER_API_KEY", ""),
            chat_model=_env_str("OPENROUTER_MODEL", "openrouter/openai/gpt-4o"),
            utility_model=_env_str(
                "OPENROUTER_UTILITY_MODEL", "openrouter/openai/gpt-4o-mini",
            ),
            base_url=_env_str(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1",
            ).rstrip("/"),
            keep_last_turns=_env_int(
                "PERSONICA_KEEP_LAST_TURNS", 5, legacy="ASSISTANT_KEEP_LAST_TURNS"),
            retrieval_top_k=_env_int(
                "PERSONICA_RETRIEVAL_TOP_K", 5, legacy="ASSISTANT_RETRIEVAL_TOP_K"),
            retrieval_min_score=_env_float(
                "PERSONICA_RETRIEVAL_MIN_SCORE", 0.20,
                legacy="ASSISTANT_RETRIEVAL_MIN_SCORE",
            ),
            relevance_weight=_env_float("PERSONICA_RELEVANCE_WEIGHT", 0.7),
            recency_half_life_days=_env_float(
                "PERSONICA_RECENCY_HALF_LIFE_DAYS", 30.0),
            timezone=_env_str(
                "PERSONICA_TIMEZONE", "Asia/Kolkata", legacy="ASSISTANT_TIMEZONE"),
            log_level=_env_str(
                "PERSONICA_LOG_LEVEL", "INFO", legacy="ASSISTANT_LOG_LEVEL"),
        )
