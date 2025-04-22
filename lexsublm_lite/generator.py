"""
LLM back‑ends (llama‑cpp‑python and Hugging Face Transformers).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from llama_cpp import Llama  # type: ignore
import torch

from .config import Settings

LOGGER = logging.getLogger("lexsublm.generator")


class BaseGenerator(ABC):
    """Abstract interface for language‑model generators."""

    @abstractmethod
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        """Return *n* completions (single‑token strings)."""

    # Optional log‑prob API for rankers
    def log_prob(self, prompt: str, token: str) -> float:  # noqa: D401
        raise NotImplementedError


class LlamaCppGenerator(BaseGenerator):
    """Local GGUF model served through llama‑cpp‑python."""

    def __init__(self, model_path: Path, n_ctx: int = 2048) -> None:
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, logits_all=True)
        self._tokenizer = self._llm.tokenize
        LOGGER.info("Loaded GGUF model: %s", model_path.name)

    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        result = self._llm(
            prompt,
            max_tokens=1,
            temperature=temperature,
            top_k=50,
            n=n,
            stop=["\n"],
        )
        return [c["text"].strip() for c in result["choices"]]

    def log_prob(self, prompt: str, token: str) -> float:
        # naïve: forward one step to get logits of first generated token
        tokens = self._tokenizer(bytes(token, "utf-8"))
        output = self._llm(prompt, max_tokens=1, logits_all=True, stop=[])
        logits = output["choices"][0]["logits"][0]  # first step
        return float(logits[tokens[0]])


class HFGenerator(BaseGenerator):
    """Quantised transformer via 🤗 Transformers + bitsandbytes."""

    def __init__(self, repo_id: str, device: str) -> None:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto" if device != "cpu" else None,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        ).eval()
        LOGGER.info("Loaded HF model: %s", repo_id)

    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        inputs = self._tokenizer([prompt] * n, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            streamer=streamer,
        )
        return [next(streamer).strip() for _ in range(n)]

    def log_prob(self, prompt: str, token: str) -> float:
        tok_id = self._tokenizer.encode(token, add_special_tokens=False)[0]
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model(**inputs)
        logits = out.logits[0, -1]  # last‑token predictions
        return float(torch.log_softmax(logits, dim=-1)[tok_id].cpu())


def build_generator() -> BaseGenerator:
    """Factory based on file extension or HF repo."""
    cfg = Settings.instance()
    path = Path(cfg.model_name)
    if path.suffix == ".gguf":
        return LlamaCppGenerator(path)
    return HFGenerator(cfg.model_name, cfg.device)
