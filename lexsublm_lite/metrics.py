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
        # If gzipped JSON, load accordingly
        if split_path.suffix == ".gz":
            import gzip, json

            with gzip.open(split_path, "rb") as fh:
                data = json.loads(fh.read())
            # Handle both nested dict format and flat list
            if isinstance(data, dict) and "substitutes" in data:
                # Nested p-lambda format
                contexts = data.get("contexts", {})
                targets = data.get("targets", {})
                subs = data.get("substitutes", {})
                labels = data.get("substitute_labels", {})
                # Map target_id -> list of substitute words
                tid_to_words: Dict[str, List[str]] = defaultdict(list)
                for sid, sub in subs.items():
                    sid_labels = labels.get(sid, [])
                    # include if any annotator marked TRUE
                    if any(lbl.startswith("TRUE") for lbl in sid_labels):
                        tid_to_words[sub["target_id"]].append(sub["substitute"])
                # Build gold map
                for tid, tgt in targets.items():
                    word = tgt.get("target")
                    key = f"{tid}::{word}"
                    self._gold[key] = tid_to_words.get(tid, [])
            elif isinstance(data, list):
                # Flat-list format: each obj has id, word, substitutes (list or pipe-string)
                for obj in data:
                    tid = obj.get("id")
                    target = obj.get("word")
                    subs = obj.get("substitutes")
                    if isinstance(subs, list):
                        words = subs
                    else:
                        words = subs.split("|") if isinstance(subs, str) else []
                    self._gold[f"{tid}::{target}"] = words
            else:
                raise ValueError(f"Unrecognized SWORDS JSON format: {type(data)}")
        else:
            # CSV path (legacy)
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
    """Adds proficiency label (A/B/C) checks for top-ranked word."""

    def __init__(self, split_path: Path):
        self._gold: Dict[str, Dict[str, str]] = defaultdict(dict)
        with open(split_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = f"{row['sent_id']}::{row['target']}"
                self._gold[key][row['substitute']] = row['level']  # A1…C2

    def prof_f1(self, gold_levels: Sequence[str], pred_level: str) -> float:
        return float(pred_level in gold_levels)

    def score_row(self, sent_id: str, target: str, pred: Sequence[str]) -> Dict[str, float]:
        gold_dict = self._gold.get(f"{sent_id}::{target}", {})
        gold_words, gold_lvls = zip(*gold_dict.items()) if gold_dict else ([], [])
        base = {
            "P@1": _ScoreHelpers.p_at_1(pred, gold_words),
            "R@10": _ScoreHelpers.recall_at_10(pred, gold_words),
            "GAP": _ScoreHelpers.gap(pred, gold_words),
        }
        if pred and gold_dict:
            base["ProF1"] = self.prof_f1(gold_lvls, gold_dict.get(pred[0], ""))
        else:
            base["ProF1"] = 0.0
        return base
