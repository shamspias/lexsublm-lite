"""
LLM wrapper for synonym generation.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from llama_cpp import Llama  # noqa: D401 – single import line
from transformers import pipeline

from .config import CACHE_DIR, DEFAULT_DEVICE, DEFAULT_MODEL

LOGGER = logging.getLogger("lexsublm.generator")
PROMPT_TEMPLATE = (
    "You are an NLP model that outputs ONE English word only.\n"
    "Replace the word **\"{target}\"** in the following sentence with a single-word "
    "synonym that preserves the original meaning and fits the context.\n"
    "Sentence: {sentence}\n\nSynonym:"
)


class LexSubGenerator:
    """Thin wrapper around a quantised causal LLM or any HF model with `text-generation`."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE) -> None:
        self.model_name = model_name
        LOGGER.info("Loading model %s on %s …", model_name, device)
        if model_name.endswith(".gguf") or model_name.endswith(".ggml"):
            self._model = Llama(model_path=model_name, n_ctx=2048, n_threads=None)
            self._type = "llama_cpp"
        else:
            self._model = pipeline(
                "text-generation", model=model_name, device=device, model_kwargs={"trust_remote_code": True}
            )
            self._type = "hf"

    @lru_cache(maxsize=8)
    def _make_prompt(self, sentence: str, target: str) -> str:
        return PROMPT_TEMPLATE.format(sentence=sentence, target=target)

    def generate(
            self,
            sentence: str,
            target: str,
            *,
            top_k: int = 5,
            temperature: float = 0.7,
            top_p: float = 0.95,
    ) -> List[str]:
        """Return *unique* one‑word candidates (length ≤ 15 chars)."""
        prompt = self._make_prompt(sentence, target)
        if self._type == "llama_cpp":
            raw = self._model(
                prompt,
                max_tokens=1,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                n=top_k,
                stop=["\n"],
                stream=False,
            )  # type: ignore[arg-type]
            texts = [o["choices"][0]["text"].strip() for o in raw]
        else:  # HF pipeline
            raw = self._model(
                prompt,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                num_return_sequences=top_k,
            )
            texts = [o["generated_text"].split()[-1].strip().strip(".,") for o in raw]

        words = {w.lower() for w in texts if 0 < len(w) <= 15 and " " not in w}
        return list(words)[: top_k]
