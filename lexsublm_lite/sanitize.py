"""
Utility functions to clean raw model outputs.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence


class Sanitizer:
    """Remove punctuation, whitespace, duplicates, sub‑words."""

    _PUNCT = set(string.punctuation)

    @staticmethod
    def clean(tokens: Iterable[str], *, max_len: int = 15) -> List[str]:
        out: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            token = t.strip().lower()
            token = token.lstrip("▁")  # sentencepiece artifacts
            if (
                    not token
                    or any(ch in Sanitizer._PUNCT for ch in token)
                    or " " in token
                    or len(token) > max_len
                    or re.fullmatch(r"\d+", token)
            ):
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
        return out
