"""
Benchmark script – writes CSV with metrics for each sentence.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from lexsublm.evaluator import LexSubEvaluator
from lexsublm.pipeline import LexSub

DATA_DIR = Path(__file__).parent / "data" / "semeval07"
GOLD_PATH = DATA_DIR / "gold.tsv"
TEST_PATH = DATA_DIR / "sentences.tsv"


def main(model: str, k: int) -> None:  # noqa: D401
    evaluator = LexSubEvaluator(GOLD_PATH)
    lexsub = LexSub(model=model)

    out_csv = Path("results") / f"{Path(model).stem}-{dt.date.today()}.csv"
    out_csv.parent.mkdir(exist_ok=True)

    aggregate: Dict[str, float] = {"P@1": 0.0, "R@10": 0.0, "GAP": 0.0}
    n = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fh_out, open(
            TEST_PATH, newline="", encoding="utf-8"
    ) as fh_in:
        writer = csv.DictWriter(fh_out, fieldnames=["sent", "target", "P@1", "R@10", "GAP"])
        writer.writeheader()
        for row in tqdm(csv.DictReader(fh_in, delimiter="\t"), desc="Evaluating"):
            sent, tgt = row["sent"], row["target"]
            preds = lexsub(sent, tgt, top_k=k)
            scores = evaluator.score(sent, tgt, preds)
            writer.writerow({"sent": sent, "target": tgt, **scores})
            for m in aggregate:
                aggregate[m] += scores[m]
            n += 1

    print("=== Aggregate ===")
    for m, val in aggregate.items():
        print(f"{m}: {val / n:.3f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/deepseek-1.5b-chat-4bit")
    p.add_argument("--k", type=int, default=10, help="Top‑k suggestions per sentence")
    main(**vars(p.parse_args()))
