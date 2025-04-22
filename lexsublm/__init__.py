"""
masked‑lexical‑substitution – top‑level API
"""

from importlib.metadata import version

from .generator import LexSubGenerator
from .pipeline import LexSub  # high‑level convenience wrapper
from .evaluator import LexSubEvaluator

__all__ = ["LexSubGenerator", "LexSubEvaluator", "LexSub", "__version__"]

__version__: str = version("masked_lexical_substitution")
