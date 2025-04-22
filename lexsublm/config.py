"""
Centralised configuration.
"""
from pathlib import Path
from typing import Final

CACHE_DIR: Final[Path] = Path.home() / ".cache" / "lexsublm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL: Final[str] = "deepseek-ai/deepseek-1.5b-chat-4bit"
DEFAULT_DEVICE: Final[str] = "cpu"
