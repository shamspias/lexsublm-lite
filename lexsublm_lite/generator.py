"""
LLM back‑ends (llama‑cpp‑python and 🤗Transformers).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from llama_cpp import Llama  # type: ignore

from .config import Settings

LOGGER = logging.getLogger("lexsublm.generator")


# --------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------- #
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]: ...

    # optional log‑prob API used by LogProbRanker
    def log_prob(self, prompt: str, token: str) -> float:  # noqa: D401
        raise NotImplementedError


# --------------------------------------------------------------------- #
# llama.cpp back‑end (GGUF files)
# --------------------------------------------------------------------- #
class LlamaCppGenerator(BaseGenerator):
    def __init__(self, model_path: Path, n_ctx: int = 2048) -> None:
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, logits_all=True)
        self._tokenize = self._llm.tokenize
        LOGGER.info("Loaded GGUF model %s", model_path)

    # -- generation ------------------------------------------------- #
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        out = self._llm(
            prompt,
            max_tokens=1,
            temperature=temperature,
            top_k=50,
            n=n,
            stop=["\n"],
        )
        return [c["text"].strip() for c in out["choices"]]

    # -- token‑level log‑prob (first position) ---------------------- #
    def log_prob(self, prompt: str, token: str) -> float:
        tok_ids = self._tokenize(token.encode())
        logits = self._llm(prompt, max_tokens=1, logits_all=True)["choices"][0]["logits"][0]
        return float(logits[tok_ids[0]])


# --------------------------------------------------------------------- #
# Transformers back‑end (4‑bit quantised)
# --------------------------------------------------------------------- #
class HFGenerator(BaseGenerator):
    def __init__(self, repo_id: str, device: str) -> None:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto" if device != "cpu" else None,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        ).eval()
        LOGGER.info("Loaded HF model %s (device=%s)", repo_id, self._model.device)

    # -- generation ------------------------------------------------- #
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        toks = self._tokenizer([prompt] * n, return_tensors="pt", padding=True)
        toks = {k: v.to(self._model.device) for k, v in toks.items()}
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._model.generate(
            **toks,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            streamer=streamer,
        )
        return [next(streamer).strip() for _ in range(n)]

    # -- log‑prob --------------------------------------------------- #
    def log_prob(self, prompt: str, token: str) -> float:
        tok_id = self._tokenizer.encode(token, add_special_tokens=False)[0]
        inp = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            logits = self._model(**inp).logits[0, -1]
        return float(torch.log_softmax(logits, dim=-1)[tok_id].cpu())


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #
def build_generator(model_name: str | None = None) -> BaseGenerator:
    cfg = Settings.instance()
    name = model_name or cfg.model_name
    path = Path(name)
    if path.suffix == ".gguf":
        return LlamaCppGenerator(path)
    return HFGenerator(name, cfg.device)
