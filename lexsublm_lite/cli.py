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
    def add_parser(sub):  # type: ignore[arg-type]
        p = sub.add_parser("run", help="Generate substitutes")
        p.add_argument("--sentence", required=True)
        p.add_argument("--target", required=True)
        p.add_argument("--top_k", type=int, default=5)
        p.add_argument("--model", help="Alias or HF repo or .gguf path")
        p.set_defaults(func=_RunCommand.run)

    @staticmethod
    def run(args):
        pipeline = LexSubPipeline(model_name=args.model)
        out = pipeline.substitute(args.sentence, args.target, k=args.top_k)
        print(json.dumps(out, ensure_ascii=False, indent=2))


class _EvalCommand:
    """Benchmark on SWORDS / ProLex."""

    @staticmethod
    def add_parser(sub: argparse._SubParsersAction[Any]) -> None:  # type: ignore[name-defined]
        p = sub.add_parser("eval", help="Evaluate on a dataset")
        p.add_argument(
            "--dataset",
            required=True,
            choices=["swords", "prolex"],
            help="Which dataset split (expects dev/test JSON/CSV in data/)",
        )
        p.add_argument("--split", default="test")
        p.set_defaults(func=_EvalCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        root = Path(__file__).resolve().parent.parent / "data"
        file_map = {
            "swords": root / "swords" / f"swords-v1.1_{args.split}.json.gz",
            "prolex": root / "prolex" / f"ProLex_v1.0_{args.split}.csv",
        }
        scorer = (
            SwordsScorer(file_map["swords"]) if args.dataset == "swords" else ProLexScorer(file_map["prolex"])
        )
        pipeline = LexSubPipeline()
        metrics_sum: Dict[str, float] = {}
        n = 0
        for sent_id, sent, target in _iter_dataset(file_map[args.dataset], args.dataset):
            preds = pipeline.substitute(sent, target)
            row = scorer.score_row(sent_id, target, preds)
            for k, v in row.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            n += 1
        means = {k: v / n for k, v in metrics_sum.items()}
        print(json.dumps(means, indent=2))


def _iter_dataset(path: Path, kind: str):
    import gzip
    import csv
    # JSON parser for nested or flat formats
    if kind == "swords":
        import orjson as json

        open_f = gzip.open if path.suffix == ".gz" else open
        with open_f(path, "rb") as fh:  # type: ignore[arg-type]
            data = json.loads(fh.read())

        # Nested p-lambda format
        if isinstance(data, dict) and "substitutes" in data:
            contexts = data.get("contexts", {})
            targets = data.get("targets", {})
            for tid, tgt in targets.items():
                # Each target has a context_id linking to contexts
                cid = tgt.get("context_id")
                ctx_entry = contexts.get(cid, {})
                sentence = None
                if isinstance(ctx_entry, dict):
                    sentence = ctx_entry.get("context")
                yield tid, sentence, tgt.get("target")

        # Flat-list format
        elif isinstance(data, list):
            for obj in data:
                yield obj["id"], obj.get("context"), obj.get("word")

        else:
            raise ValueError(f"Unrecognized SWORDS JSON format: {type(data)}")
    else:
        # ProLex CSV format
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row["sent_id"], row["sentence"], row["target"]


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(prog="lexsub")
    subs = parser.add_subparsers(dest="command", required=True)
    _RunCommand.add_parser(subs)
    _EvalCommand.add_parser(subs)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
