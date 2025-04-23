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
        return len(set(pred[:10]) & set(gold)) / len(gold) if gold else 0.0

    @staticmethod
    def gap(pred: Sequence[str], gold: Sequence[str]) -> float:
        hits, gap_sum = 0, 0.0
        for i, w in enumerate(pred, 1):
            if w in gold:
                hits += 1
                gap_sum += hits / i
        return gap_sum / len(gold) if gold else 0.0


class SwordsScorer:
    """Compute mean P@1 / R@10 / GAP over SWORDS dev|test splits."""

    def __init__(self, split_path: Path):
        self._gold: Dict[str, List[str]] = {}
        # If gzipped JSON, load accordingly
        if split_path.suffix == ".gz":
            import gzip, json

            with gzip.open(split_path, "rb") as fh:
                data = json.loads(fh.read())
            # Handle both nested dict format and flat list
            if isinstance(data, dict) and "substitutes" in data:
                contexts = data.get("contexts", {})
                targets = data.get("targets", {})
                subs = data.get("substitutes", {})
                labels = data.get("substitute_labels", {})
                tid_to_words: Dict[str, List[str]] = defaultdict(list)
                for sid, sub in subs.items():
                    sid_labels = labels.get(sid, [])
                    if any(lbl.startswith("TRUE") for lbl in sid_labels):
                        tid_to_words[sub["target_id"]].append(sub["substitute"])
                for tid, tgt in targets.items():
                    word = tgt.get("target")
                    key = f"{tid}::{word}"
                    self._gold[key] = tid_to_words.get(tid, [])
            elif isinstance(data, list):
                for obj in data:
                    tid = obj.get("id")
                    target = obj.get("word")
                    subs = obj.get("substitutes")
                    words = subs if isinstance(subs, list) else (subs.split("|") if isinstance(subs, str) else [])
                    self._gold[f"{tid}::{target}"] = words
            else:
                raise ValueError(f"Unrecognized SWORDS JSON format: {type(data)}")
        else:
            with open(split_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = f"{row['sent_id']}::{row['target']}"
                    self._gold[key] = row['substitutes'].split("|")

    def score_row(self, sent_id: str, target: str, pred: Sequence[str]) -> Dict[str, float]:
        gold = self._gold.get(f"{sent_id}::{target}", [])
        return {
            "P@1": _ScoreHelpers.p_at_1(pred, gold),
            "R@10": _ScoreHelpers.recall_at_10(pred, gold),
            "GAP": _ScoreHelpers.gap(pred, gold),
        }


class ProLexScorer:
    """
    Compute P@1 / R@10 / GAP and ProF1 over ProLex’s proficiency-oriented substitutes.

    The ProLex CSV has columns:
      - target word
      - Sentence
      - prof_acc_subs (a Python-list string of advanced substitutes)

    We enumerate rows to assign sent_ids internally.
    """

    def __init__(self, split_path: Path):
        import ast
        self._gold: Dict[str, List[str]] = {}
        with open(split_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                sent_id = str(idx)
                target = row.get("target word")
                prof_list: List[str] = []
                if row.get("prof_acc_subs"):
                    prof_list = ast.literal_eval(row["prof_acc_subs"])
                self._gold[f"{sent_id}::{target}"] = prof_list

    def score_row(self, sent_id: str, target: str, pred: Sequence[str]) -> Dict[str, float]:
        gold = self._gold.get(f"{sent_id}::{target}", [])
        p1 = _ScoreHelpers.p_at_1(pred, gold)
        r10 = _ScoreHelpers.recall_at_10(pred, gold)
        gap = _ScoreHelpers.gap(pred, gold)
        prof1 = float(bool(pred and pred[0] in gold))
        return {"P@1": p1, "R@10": r10, "GAP": gap, "ProF1": prof1}
