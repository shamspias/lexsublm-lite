"""
CLI with `run`, `eval`, and `download` sub-commands (argparse, class-based).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from .metrics import ProLexScorer, SwordsScorer
from .pipeline import LexSubPipeline

_LOG = logging.getLogger("lexsublm.cli")


class _RunCommand:
    @staticmethod
    def add_parser(sub: argparse._SubParsersAction) -> None:
        p = sub.add_parser("run", help="Generate substitutes")
        p.add_argument("--sentence", required=True, help="Input sentence")
        p.add_argument("--target", required=True, help="Target word to substitute")
        p.add_argument("--top_k", type=int, default=5, help="Number of substitutes to return")
        p.add_argument("--model", help="Alias or HF repo or .gguf path (overrides default)")
        p.set_defaults(func=_RunCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        pipeline = LexSubPipeline(model_name=args.model)
        out = pipeline.substitute(args.sentence, args.target, k=args.top_k)
        print(json.dumps(out, ensure_ascii=False, indent=2))


class _EvalCommand:
    """Benchmark on SWORDS / ProLex."""

    @staticmethod
    def add_parser(sub: argparse._SubParsersAction[Any]) -> None:
        p = sub.add_parser("eval", help="Evaluate on a dataset")
        p.add_argument("--dataset",
                       required=True,
                       choices=["swords", "prolex"],
                       help="Which dataset to evaluate: swords or prolex")
        p.add_argument("--split",
                       default="test",
                       choices=["dev", "test"],
                       help="Which split to use (dev or test)")
        p.add_argument("--model", help="Alias or HF repo or .gguf path (overrides default)")
        p.set_defaults(func=_EvalCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        root = Path(__file__).resolve().parent.parent / "data"
        file_map: Dict[str, Path] = {
            "swords": root / "swords" / f"swords-v1.1_{args.split}.json.gz",
            "prolex": root / "prolex" / f"ProLex_v1.0_{args.split}.csv",
        }
        # Initialize scorer based on dataset
        if args.dataset == "swords":
            scorer = SwordsScorer(file_map["swords"])
        else:
            scorer = ProLexScorer(file_map["prolex"])

        # Build pipeline, optionally with a custom model
        pipeline = LexSubPipeline(model_name=args.model)

        metrics_sum: Dict[str, float] = {}
        n = 0
        # Iterate through dataset
        for sent_id, sent, target in _iter_dataset(file_map[args.dataset], args.dataset):
            preds = pipeline.substitute(sent, target)
            row = scorer.score_row(sent_id, target, preds)
            for metric, value in row.items():
                metrics_sum[metric] = metrics_sum.get(metric, 0.0) + value
            n += 1

        # Compute means
        means = {metric: total / n for metric, total in metrics_sum.items()}
        print(json.dumps(means, indent=2))


def _iter_dataset(path: Path, kind: str):
    import gzip
    import csv

    if kind == "swords":
        import orjson as json
        open_fn = gzip.open if path.suffix == ".gz" else open
        with open_fn(path, "rb") as fh:
            data = json.loads(fh.read())
        if isinstance(data, dict) and "substitutes" in data:
            contexts = data.get("contexts", {})
            targets = data.get("targets", {})
            for tid, tgt in targets.items():
                cid = tgt.get("context_id")
                ctx = contexts.get(cid, {})
                sentence = ctx.get("context") if isinstance(ctx, dict) else None
                yield tid, sentence, tgt.get("target")
        elif isinstance(data, list):
            for obj in data:
                yield obj.get("id"), obj.get("context"), obj.get("word")
        else:
            raise ValueError(f"Unrecognized SWORDS JSON format: {type(data)}")
    else:
        # ProLex CSV
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row["sent_id"], row["sentence"], row["target"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lexsub")
    subs = parser.add_subparsers(dest="command", required=True)
    _RunCommand.add_parser(subs)
    _EvalCommand.add_parser(subs)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
