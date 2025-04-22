"""
Official SemEval‑2007 metrics: P@1, Recall@10, GAP.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

from .filter import NLP


class LexSubEvaluator:
    """Compute metrics against gold substitutes."""

    def __init__(self, gold_path: Path):
        self.gold = self._load_gold(gold_path)

    @staticmethod
    def _normalize(word: str) -> str:
        return NLP(word.lower())[0].lemma_

    def _load_gold(self, path: Path) -> Dict[str, List[str]]:
        gold: Dict[str, List[str]] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                key = f"{row['sent']}\t{row['target']}"
                subs = [self._normalize(w) for w in row["substitutes"].split(",")]
                gold[key] = subs
        return gold

    def score(self, sentence: str, target: str, preds: Sequence[str]) -> Dict[str, float]:
        key = f"{sentence}\t{target}"
        gold = self.gold[key]
        norm_preds = [self._normalize(p) for p in preds]

        p1 = 1.0 if norm_preds and norm_preds[0] in gold else 0.0
        rec10 = len(set(norm_preds[:10]) & set(gold)) / len(gold)
        gap = self._gap(gold, norm_preds)
        return {"P@1": p1, "R@10": rec10, "GAP": gap}

    @staticmethod
    def _gap(gold: List[str], preds: Sequence[str]) -> float:
        """Generalised Average Precision."""
        hits = 0
        score = 0.0
        for i, p in enumerate(preds, 1):
            if p in gold:
                hits += 1
                score += hits / i
        return score / len(gold) if gold else 0.0
