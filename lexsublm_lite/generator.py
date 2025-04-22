"""
Model loader that supports:
• Hugging Face repos (full / 4‑bit)      – CPU / MPS / CUDA
• GGUF files via llama.cpp               – all OSes
• Friendly aliases via model_registry.yaml
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import ClassVar, Dict, Sequence
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

try:
    from transformers import BitsAndBytesConfig  # noqa: WPS433

    HAVE_BNB = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore
    HAVE_BNB = False

from llama_cpp import Llama  # type: ignore
from .config import Settings

_LOG = logging.getLogger("lexsublm.generator")

# ---------- registry helper ------------------------------------------------ #
_REG_PATH = Path(__file__).resolve().parent.parent / "model_registry.yaml"
_ALIAS: Dict[str, str] = yaml.safe_load(_REG_PATH.read_text()) if _REG_PATH.exists() else {}


def _resolve(name: str) -> str:
    """Turn alias → repo‑id / path if defined in registry."""
    return _ALIAS.get(name, name)


# ---------- abstract base --------------------------------------------------- #
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]: ...

    def log_prob(self, prompt: str, token: str) -> float: ...  # optional


# ---------- llama.cpp (GGUF) ------------------------------------------------ #
class LlamaCppGen(BaseGenerator):
    def __init__(self, path: Path, n_ctx: int = 2048) -> None:
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, logits_all=True)
        self._tok = self.llm.tokenize
        _LOG.info("GGUF loaded: %s", path.name)

    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        out = self.llm(prompt, max_tokens=1, n=n, temperature=temperature, top_k=50, stop=["\n"])
        return [c["text"].strip() for c in out["choices"]]

    def log_prob(self, prompt: str, token: str) -> float:
        tid = self._tok(token.encode())[0]
        logits = self.llm(prompt, max_tokens=1, logits_all=True)["choices"][0]["logits"][0]
        return float(logits[tid])


# ---------- HF (full / 4‑bit) ---------------------------------------------- #
class HFGen(BaseGenerator):
    _cache: ClassVar[Dict[str, "HFGen"]] = {}

    def __new__(cls, repo: str, device: str):
        key = f"{repo}-{device}"
        if key in cls._cache:
            return cls._cache[key]
        inst = super().__new__(cls)
        cls._cache[key] = inst
        return inst

    def __init__(self, repo: str, device: str) -> None:
        if getattr(self, "_ready", False):
            return
        self._ready = True

        quant_cfg = None
        if device == "cuda" and HAVE_BNB:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            _LOG.info("4‑bit bitsandbytes active.")
        else:
            _LOG.info("Full / 8‑bit (device=%s) – bitsandbytes disabled.", device)

        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            device_map="auto" if device != "cpu" else None,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            trust_remote_code=True,
            quantization_config=quant_cfg,
        ).eval()
        self.device = self.model.device
        _LOG.info("Model ready on %s", self.device)

    # -- generation ------------------------------------------------- #
    def generate(self, prompt: str, *, n: int = 5, temperature: float = 0.7) -> Sequence[str]:
        if n == 1:
            # streaming path (nice for inter­active UIs, still batch=1)
            toks = self.tok(prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            self.model.generate(
                **toks,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                streamer=streamer,
            )
            return [next(streamer).strip()]
        # ---------- n > 1 ------------------------------------------------ #
        out: list[str] = []
        for _ in range(n):
            toks = self.tok(prompt, return_tensors="pt").to(self.device)
            gen_ids = self.model.generate(
                **toks,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                pad_token_id=self.tok.eos_token_id,
            )[0]
            # last token is the new word
            out.append(self.tok.decode(gen_ids[-1]).strip())
        return out

    # -- log‑prob ---------------------------------------------- #
    def log_prob(self, prompt: str, token: str) -> float:
        tid = self.tok.encode(token, add_special_tokens=False)[0]
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inp).logits[0, -1]
        return float(torch.log_softmax(logits, dim=-1)[tid].cpu())


# ---------- factory -------------------------------------------------------- #
def build_generator(name: str | None = None) -> BaseGenerator:
    cfg = Settings.instance()
    resolved = _resolve(name or cfg.model_name)
    path = Path(resolved)
    if path.suffix == ".gguf":
        return LlamaCppGen(path)
    return HFGen(resolved, cfg.device)
