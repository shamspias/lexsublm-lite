"""
Centralised configuration, validated by Pydantic   v3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment‑driven configuration."""

    # ---- public fields (can be overridden via env or CLI) ---------------- #
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
    device: str = "cpu"  # "mps" for Apple   Silicon, "cuda" for GPU
    cache_dir: Path = Path.home() / ".cache" / "lexsublm_lite"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="LEXSUB_", case_sensitive=False)

    # ---- lightweight singleton helper (ignored by pydantic) -------------- #
    _instance: ClassVar[Optional["Settings"]] = None  # noqa: RUF012

    @classmethod
    def instance(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = cls()  # create & cache exactly once
            logging.basicConfig(
                level=getattr(logging, cls._instance.log_level.upper(), logging.INFO),
                format="%(levelname)s | %(name)s | %(message)s",
            )
        return cls._instance
