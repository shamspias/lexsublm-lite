"""
Centralised configuration, validated by Pydantic v3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment‑driven configuration."""

    model_name: str = "zamal/Deepseek-VL-1.3b-chat-4bit"
    device: str = "cpu"
    cache_dir: Path = Path.home() / ".cache" / "lexsublm_lite"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="LEXSUB_", case_sensitive=False)

    # ---- singleton helper (ignored by pydantic because of ClassVar) ---- #
    _instance: ClassVar[Optional["Settings"]] = None

    @classmethod
    def instance(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = cls()  # create the settings model **once**
            logging.basicConfig(
                level=getattr(logging, cls._instance.log_level.upper(), logging.INFO),
                format="%(levelname)s | %(name)s | %(message)s",
            )
        return cls._instance
