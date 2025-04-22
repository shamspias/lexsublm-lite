"""
High‑level convenience wrapper: generate → sanitize → filter → rank.
"""
from __future__ import annotations

import logging
from typing import List

from .config import Settings
from .filter import CosineFilter, PosMorphFilter
from .generator import BaseGenerator, build_generator
from .ranker import LogProbRanker, SBertRanker
from .sanitize import Sanitizer

LOGGER = logging.getLogger("lexsublm.pipeline")


class LexSubPipeline:
    """Single entry‑point used by CLI & notebooks."""

    def __init__(
            self,
            *,
            generator: BaseGenerator | None = None,
            use_sbert_ranker: bool = True,
    ) -> None:
        self.cfg = Settings.instance()
        self.gen = generator or build_generator()
        self.filters = [PosMorphFilter(), CosineFilter()]
        self.ranker = SBertRanker() if use_sbert_ranker else LogProbRanker(self.gen)

    def substitute(self, sentence: str, target: str, *, k: int = 5) -> List[str]:
        prompt = (
            f"Return ONE English word that can replace '{target}' in this sentence "
            f"so that the new sentence keeps exactly the same meaning.\nSentence: {sentence}\nSynonym:"
        )
        raw = self.gen.generate(prompt, n=max(20, k * 4))
        cands = Sanitizer.clean(raw)
        for f in self.filters:
            cands = f(target, cands)
        return self.ranker.rank(sentence, target, cands=cands, top_k=k)
