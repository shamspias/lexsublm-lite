"""
Command‑line interface – `python -m lexsublm.cli ...`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import LexSub


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Context‑aware lexical substitution.")
    parser.add_argument("--sentence", required=True, help="Input sentence")
    parser.add_argument("--target", required=True, help="Target word to replace")
    parser.add_argument("--top_k", type=int, default=5, help="Number of suggestions")
    parser.add_argument("--model", help="HF repo or GGUF path (default: DeepSeek‑1.5B 4‑bit)")
    args = parser.parse_args(argv)

    lexsub = LexSub(model=args.model)
    preds = lexsub(args.sentence, args.target, top_k=args.top_k)
    print(json.dumps(preds, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
