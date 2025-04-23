"""
CLI with `run` and `eval` sub-commands (argparse, class-based).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from .metrics import ProLexScorer, SwordsScorer, TsarScorer
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
    """Benchmark on SWORDS / ProLex / TSAR-2022."""

    @staticmethod
    def add_parser(sub: argparse._SubParsersAction[Any]) -> None:
        p = sub.add_parser("eval", help="Evaluate on a dataset")
        p.add_argument(
            "--dataset",
            required=True,
            choices=["swords", "prolex", "tsar22"],
            help="Which dataset to evaluate: swords, prolex, or tsar22",
        )
        p.add_argument(
            "--split",
            required=True,
            help=(
                "Which split to use:\n"
                "  - swords/prolex: 'dev' or 'test'\n"
                "  - tsar22: 'test' (alias for test_none), 'test_none', or 'test_gold'"
            ),
        )
        p.add_argument(
            "--model",
            help="Alias or HF repo or .gguf path (overrides default)",
        )
        p.set_defaults(func=_EvalCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        root = Path(__file__).resolve().parent.parent / "data"

        # Select scorer & data path
        if args.dataset == "swords":
            split_path = root / "swords" / f"swords-v1.1_{args.split}.json.gz"
            scorer = SwordsScorer(split_path)
        elif args.dataset == "prolex":
            split_path = root / "prolex" / f"ProLex_v1.0_{args.split}.csv"
            scorer = ProLexScorer(split_path)
        else:  # tsar22
            # map 'test' to 'test_none'
            if args.split == "test":
                split_name = "test_none"
            elif args.split in ("test_none", "test_gold"):
                split_name = args.split
            else:
                raise ValueError(
                    "For tsar22, --split must be 'test', 'test_none', or 'test_gold'"
                )
            # currently hard-coded to English; extend with --lang if needed
            split_path = root / "tsar22" / f"tsar2022_en_{split_name}.tsv"
            scorer = TsarScorer(split_path)

        # Initialize pipeline
        pipeline = LexSubPipeline(model_name=args.model)

        # Aggregate metrics
        metrics_sum: Dict[str, float] = {}
        n = 0
        for sent_id, sent, target in _iter_dataset(split_path, args.dataset):
            preds = pipeline.substitute(sent, target)
            row = scorer.score_row(sent_id, target, preds)
            for metric, value in row.items():
                metrics_sum[metric] = metrics_sum.get(metric, 0.0) + value
            n += 1

        # Print means
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
        else:
            for obj in data:  # list format
                yield obj.get("id"), obj.get("context"), obj.get("word")

    elif kind == "prolex":
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                sent_id = str(idx)
                sentence = row.get("Sentence") or row.get("sentence")
                target = row.get("target word")
                yield sent_id, sentence, target

    else:  # tsar22
        # TSV format: Sentence<TAB>ComplexWord<TAB>...
        with open(path, newline="", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                sentence, target = parts[0], parts[1]
                yield str(idx), sentence, target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lexsub")
    subs = parser.add_subparsers(dest="command", required=True)
    _RunCommand.add_parser(subs)
    _EvalCommand.add_parser(subs)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
