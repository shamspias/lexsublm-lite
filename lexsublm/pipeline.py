"""
End‑to‑end convenience class: generate → filter → rank.
"""
from __future__ import annotations

from typing import List

from .filter import CosineFilter, PosFilter
from .generator import LexSubGenerator
from .ranker import SBertRanker


class LexSub:
    """High‑level helper used by CLI & notebooks."""

    def __init__(self, model: str | None = None) -> None:
        self.generator = LexSubGenerator(model)  # noqa: D401 – simple init
        self.filters = [PosFilter(), CosineFilter()]
        self.ranker = SBertRanker()

    def __call__(self, sentence: str, target: str, *, top_k: int = 5) -> List[str]:
        cands = self.generator.generate(sentence, target, top_k=top_k * 2)
        for f in self.filters:
            cands = f(sentence, target, cands)
        ranked = self.ranker(sentence, target, cands)
        return ranked[:top_k]
