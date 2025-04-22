"""
Ranking strategies for candidate synonyms.
"""
from __future__ import annotations

from typing import List, Sequence

from sentence_transformers import SentenceTransformer, util

SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class SBertRanker:
    """Rank by cosine similarity between substitute & full sentence (target replaced)."""

    def __call__(self, sentence: str, target: str, candidates: Sequence[str]) -> List[str]:
        base = sentence.replace(target, "{}")
        sent_emb = SBERT.encode(sentence, convert_to_tensor=True)
        scored = []
        for c in candidates:
            new_sent = base.format(c)
            sim = util.cos_sim(sent_emb, SBERT.encode(new_sent, convert_to_tensor=True)).item()
            scored.append((sim, c))
        return [c for _, c in sorted(scored, key=lambda t: t[0], reverse=True)]
