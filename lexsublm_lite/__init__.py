"""
LexSubLM‑Lite – public interface
"""
from __future__ import annotations

from importlib.metadata import version

from .pipeline import LexSubPipeline

__all__ = ["LexSubPipeline", "__version__"]

__version__: str = version("lexsublm_lite")
