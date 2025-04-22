"""
Ranking strategies: log‑probability or embedding similarity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from sentence_transformers import SentenceTransformer, util

from .generator import BaseGenerator
from .sanitize import Sanitizer


class BaseRanker(ABC):
    """Return candidates ordered best‑to‑worst."""

    @abstractmethod
    def rank(
            self,
            sentence: str,
            target: str,
            *,
            cands: Sequence[str],
            top_k: int,
    ) -> List[str]: ...


class LogProbRanker(BaseRanker):
    """Score using the generator’s own next‑token log‑prob."""

    def __init__(self, gen: BaseGenerator) -> None:
        self._gen = gen

    def rank(
            self,
            sentence: str,
            target: str,
            *,
            cands: Sequence[str],
            top_k: int = 5,
    ) -> List[str]:
        prompt = (
            f"You are a helpful NLP model. Replace the word '{target}' in the following sentence "
            f"with a synonym that fits perfectly.\nSentence: {sentence}\nSynonym:"
        )
        scored = [(self._gen.log_prob(prompt, w), w) for w in cands]
        return [w for _, w in sorted(scored, key=lambda t: t[0], reverse=True)][:top_k]


class SBertRanker(BaseRanker):
    """Rank by change‑in‑sentence cosine with e5‑small‑v2."""

    _model = SentenceTransformer("intfloat/e5-small-v2")

    def rank(
            self,
            sentence: str,
            target: str,
            *,
            cands: Sequence[str],
            top_k: int = 5,
    ) -> List[str]:
        template = sentence.replace(target, "{}")
        base_vec = self._model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
        scored = []
        for w in cands:
            new_vec = self._model.encode(template.format(w), convert_to_tensor=True, normalize_embeddings=True)
            scored.append((float(util.dot_score(base_vec, new_vec)), w))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [w for _, w in scored][:top_k]
