"""
Quick benchmark for LexSubLM‑Lite models
────────────────────────────────────────
* Discovers aliases from model_registry.yaml
* Evaluates on 5 hand‑crafted gold test‑cases
* Prints P@1, Recall@k, Jaccard using tabulate2

Run:
    python -m lexsublm_lite.bench.bench_models --top_k 5
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import Dict, List

from tabulate2 import tabulate  # pip install tabulate2
from lexsublm_lite.pipeline import LexSubPipeline

# ───────────────────────── GOLD TEST‑SET (5 cases) ─────────────────────── #
TESTS = [
    {
        "sentence": "The bright student aced the exam.",
        "target": "bright",
        "gold": ["brilliant", "smart", "gifted", "clever", "talented"],
    },
    {
        "sentence": "He sat on the bank of the river.",
        "target": "bank",
        "gold": ["shore", "riverbank", "waterside", "embankment", "edge"],
    },
    {
        "sentence": "A cold wind blew across the field.",
        "target": "cold",
        "gold": ["chilly", "cool", "frigid", "freezing", "icy"],
    },
    {
        "sentence": "The fast car sped down the highway.",
        "target": "fast",
        "gold": ["quick", "speedy", "rapid", "swift", "accelerated"],
    },
    {
        "sentence": "The happy child laughed loudly.",
        "target": "happy",
        "gold": ["joyful", "cheerful", "content", "glad", "delighted"],
    },
]


# ───────────────────────────── METRICS ──────────────────────────────── #
def p_at_1(pred: List[str], gold: List[str]) -> float:
    return float(bool(pred and pred[0] in gold))


def recall_at_k(pred: List[str], gold: List[str], k: int) -> float:
    return len(set(pred[:k]) & set(gold)) / len(gold)


def jaccard(pred: List[str], gold: List[str]) -> float:
    s1, s2 = set(pred), set(gold)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0


# ───────────────────────────── BENCHMARK ────────────────────────────── #
def load_registry() -> Dict[str, str]:
    reg_path = Path(__file__).resolve().parent.parent.parent / "model_registry.yaml"
    if not reg_path.exists():
        raise FileNotFoundError("model_registry.yaml not found.")
    return yaml.safe_load(reg_path.read_text())


def bench_alias(alias: str, k: int) -> Dict[str, float]:
    pipe = LexSubPipeline(model_name=alias)
    totals = {"P@1": 0.0, f"R@{k}": 0.0, "Jaccard": 0.0}
    for case in TESTS:
        preds = pipe.substitute(case["sentence"], case["target"], k=k)
        totals["P@1"] += p_at_1(preds, case["gold"])
        totals[f"R@{k}"] += recall_at_k(preds, case["gold"], k)
        totals["Jaccard"] += jaccard(preds, case["gold"])
    n = len(TESTS)
    return {m: totals[m] / n for m in totals}


def main() -> None:  # noqa: D401
    ap = argparse.ArgumentParser(description="Benchmark LexSubLM‑Lite models")
    ap.add_argument("--top_k", type=int, default=5, help="Top‑k suggestions to evaluate")
    args = ap.parse_args()

    registry = load_registry()
    rows: List[List[str]] = []
    for alias in registry:
        print(f"→ running {alias}")
        try:
            scores = bench_alias(alias, args.top_k)
            rows.append([
                alias,
                f"{scores['P@1']:.2f}",
                f"{scores[f'R@{args.top_k}']:.2f}",
                f"{scores['Jaccard']:.2f}",
            ])
        except Exception as e:
            print(f"⚠️  {alias} failed: {e.__class__.__name__}: {e}")
            rows.append([alias, "ERR", "ERR", "ERR"])

    # sort by Precision@1 descending
    rows.sort(key=lambda r: float(r[1]) if r[1] not in {"ERR"} else -1, reverse=True)
    print("\n" + tabulate(rows, headers=["Model", "P@1", f"R@{args.top_k}", "Jaccard"], tablefmt="github"))


if __name__ == "__main__":
    main()
