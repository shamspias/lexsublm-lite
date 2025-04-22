"""
Official SWORDS + ProLex metrics (P@1, R@10, GAP, Proficiency‑F1).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


class _ScoreHelpers:
    """Static helpers not exposed publicly."""

    @staticmethod
    def p_at_1(pred: Sequence[str], gold: Sequence[str]) -> float:
        return float(bool(pred and pred[0] in gold))

    @staticmethod
    def recall_at_10(pred: Sequence[str], gold: Sequence[str]) -> float:
        return len(set(pred[:10]) & set(gold)) / len(gold)

    @staticmethod
    def gap(pred: Sequence[str], gold: Sequence[str]) -> float:
        hits, gap_sum = 0, 0.0
        for i, w in enumerate(pred, 1):
            if w in gold:
                hits += 1
                gap_sum += hits / i
        return gap_sum / len(gold)


class SwordsScorer:
    """Compute mean P@1 / R@10 / GAP over SWORDS dev|test splits."""

    def __init__(self, split_path: Path):
        self._gold: Dict[str, List[str]] = {}
        with open(split_path, newline="", encoding="utf-8") as fh:
            data = csv.DictReader(fh)
            for row in data:
                key = f"{row['sent_id']}::{row['target']}"
                self._gold[key] = row["substitutes"].split("|")

    def score_row(self, sent_id: str, target: str, pred: Sequence[str]) -> Dict[str, float]:
        gold = self._gold[f"{sent_id}::{target}"]
        return {
            "P@1": _ScoreHelpers.p_at_1(pred, gold),
            "R@10": _ScoreHelpers.recall_at_10(pred, gold),
            "GAP": _ScoreHelpers.gap(pred, gold),
        }


class ProLexScorer:
    """Adds proficiency label (A/B/C) checks for top‑ranked word."""

    def __init__(self, split_path: Path):
        self._gold: Dict[str, Dict[str, str]] = defaultdict(dict)
        with open(split_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = f"{row['sent_id']}::{row['target']}"
                self._gold[key][row["substitute"]] = row["level"]  # A1…C2

    def prof_f1(self, gold_levels: Sequence[str], pred_level: str) -> float:
        return float(pred_level in gold_levels)

    def score_row(self, sent_id: str, target: str, pred: Sequence[str]) -> Dict[str, float]:
        gold_dict = self._gold[f"{sent_id}::{target}"]
        gold_words, gold_lvls = zip(*gold_dict.items())
        base = {
            "P@1": _ScoreHelpers.p_at_1(pred, gold_words),
            "R@10": _ScoreHelpers.recall_at_10(pred, gold_words),
            "GAP": _ScoreHelpers.gap(pred, gold_words),
        }
        if pred:
            base["ProF1"] = self.prof_f1(gold_lvls, gold_dict.get(pred[0], ""))
        else:
            base["ProF1"] = 0.0
        return base
