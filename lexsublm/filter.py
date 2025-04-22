"""
Filters for generated substitutes.
"""
from __future__ import annotations

from typing import Iterable, List

import spacy
from sentence_transformers import SentenceTransformer, util

NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class PosFilter:
    """Keep only candidates with the same coarse POS tag (noun, verb, adj, adv)."""

    def __call__(self, sentence: str, target: str, candidates: Iterable[str]) -> List[str]:
        target_pos = NLP(target)[0].pos_
        return [c for c in candidates if NLP(c)[0].pos_ == target_pos]


class CosineFilter:
    """Discard candidates whose SBERT cosine similarity with target < threshold."""

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def __call__(self, sentence: str, target: str, candidates: Iterable[str]) -> List[str]:
        tg_emb = SBERT.encode(target, convert_to_tensor=True)
        out: List[str] = []
        for c in candidates:
            sim = util.cos_sim(tg_emb, SBERT.encode(c, convert_to_tensor=True)).item()
            if sim >= self.threshold:
                out.append(c)
        return out
