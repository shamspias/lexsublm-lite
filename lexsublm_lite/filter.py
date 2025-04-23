"""
Candidate filters (POS + morphological agreement, cosine similarity).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import spacy
from sentence_transformers import SentenceTransformer, util

_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
# _SBERT = SentenceTransformer("intfloat/e5-small-v2")  # compact 20 MB int8
_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _coarse_tag(word: str) -> str:
    return _NLP(word)[0].pos_  # inexpensive (single token)


class PosMorphFilter:
    """Keep substitutes with identical coarse POS & morphology."""

    def __call__(self, target: str, cands: Iterable[str]) -> List[str]:
        tgt_tok = _NLP(target)[0]
        return [
            c
            for c in cands
            if (tok := _NLP(c)[0]).pos_ == tgt_tok.pos_ and tok.morph.get("Number") == tgt_tok.morph.get("Number")
        ]


class CosineFilter:
    """Discard substitutes having SBERT cosine < threshold against target."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.th = threshold

    @lru_cache(maxsize=1_024)
    def _embed(self, text: str):
        return _SBERT.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    def __call__(self, target: str, cands: Iterable[str]) -> List[str]:
        tgt_vec = self._embed(target)
        out: list[str] = []
        for w in cands:
            if float(util.dot_score(tgt_vec, self._embed(w))) >= self.th:
                out.append(w)
        return out
