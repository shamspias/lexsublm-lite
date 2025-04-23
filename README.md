# LexSubLM-Lite

*Fast, context-aware lexical substitution that **really** fits on a laptop.*

---

## 1 · What is it?

LexSubLM-Lite is a **Python toolkit** for proposing single-word substitutes that preserve both the meaning **and** syntax of a target word within its sentence. It is ideal for NLP applications that require controlled synonym generation without heavy dependencies.

## 2 · Why use LexSubLM-Lite?

- **Test new models quickly**: Swap in your own models—HF repos or local GGUF quantisations—to see how they perform on standard lexical substitution tasks.
- **Lightweight research**: Run benchmarks on your laptop CPU or GPU without needing large server infrastructure.
- **Easy model extension**: Add new generators by editing `model_registry.yaml`, no code changes needed.
- **Reproducible results**: Dockerfile and helper scripts enable one-command setup and dataset downloads.
- **Research-grade metrics**: Built-in evaluation scripts output P@1, Recall@k, GAP, and ProLex Pro-F1 for rigorous analysis.

## 3 · Key features

| Stage                   | What we do                                                                     | Why it matters                                            |
|-------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------|
| **Prompted generation** | Causal LLM (4-bit when CUDA, fp16/fp32 otherwise) returns *k* candidates.      | Runs on laptop CPU or GPU.                                |
| **Sanitisation**        | Strips punctuation / multi-word babble.                                        | Keeps outputs clean.                                      |
| **POS + morph filter**  | spaCy + pymorphy3 — keeps tense, number, degree.                               | Prevents form errors (e.g., “cats → cat”).                |
| **Ranking**             | Log-prob *or* cosine with `e5-small-v2` (<40 MB).                              | Balance quality vs. footprint.                            |
| **Evaluation**          | P@1, Recall@k, GAP, ProF1 (optional).                                          | Research-grade metrics.                                   |
| **Model registry**      | `model_registry.yaml` maps *alias → HF repo / GGUF path*.                      | Add models without touching code.                         |
| **Benchmark script**    | `python -m lexsublm_lite.bench.bench_models` prints a table via **tabulate2**. | One-shot comparison across aliases.                       |

## 4 · Install

```bash
# 1. Create & activate a virtualenv or conda environment
python -m pip install --upgrade pip

# 2. Clone & install in development mode
git clone https://github.com/shamspias/lexsublm-lite
cd lexsublm-lite
pip install -e .
```

> **CUDA users**: install `bitsandbytes` to enable true 4-bit quant (`pip install bitsandbytes`).

## 5 · Quick start (run)

Generate top-k substitutes:

```bash
lexsub run \
  --sentence "The bright student aced the exam." \
  --target bright \
  --top_k 5 \
  --model llama3-mini
```

<details>
<summary>Example JSON output</summary>

```json
[
  "brilliant",
  "smart",
  "gifted",
  "clever",
  "talented"
]
```

</details>

### Toggle model source

```bash
# By alias
lexsub run ... --model distilgpt2

# Direct HF repo
lexsub run ... --model EleutherAI/gpt-neo-125m

# Local GGUF (fastest on macOS)
lexsub run ... --model ./models/MyQuantizedModel.gguf
```

Run a mini-benchmark on 5 hand‑crafted cases across all registry aliases:

```bash
python -m lexsublm_lite.bench.bench_models --top_k 5
```

## 6 · Evaluation (eval)

Benchmark any model on standard datasets (SWORDS, ProLex, TSAR-2022) and output aggregate metrics:

```bash
lexsub eval \
  --dataset swords \
  --split dev \
  --model llama3-mini
```

- `--dataset`: `swords` | `prolex` | `tsar22`
- `--split`:
  - **swords/prolex**: `dev` or `test`
  - **tsar22**: `test` (alias for `test_none`), `test_none`, or `test_gold`
- `--model`: alias, HF repo, or `.gguf` path (overrides default)

The command prints mean P@1, Recall@k, GAP (and ProF1 for ProLex).

## 7 · Datasets

| Corpus             | Download helper                                        | Size                               | License   |
|--------------------|--------------------------------------------------------|------------------------------------|-----------|
| **SWORDS (2021)**  | `python -m lexsub.datasets.swords.download`            | 4 848 targets / 57 k subs          | CC-BY-4.0 |
| **ProLex (2024)**  | `python -m lexsub.datasets.prolex.download`            | 6 000 sentences + proficiency ranks| CC-BY-4.0 |
| **TSAR-2022**      | `python -m lexsub.datasets.tsar.download`              | EN/ES/PT – 1 133 sents             | CC-BY-4.0 |
| **SemEval-2007**   | (legacy)                                               | 2 000 sents                        | CC-BY-2.5 |

## 8 · Default model registry

```yaml
llama3-mini: meta-llama/Llama-3.2-1B

distilgpt2: distilbert/distilgpt2

qwen500m: Qwen/Qwen2.5-0.5B

tinyllama: Maykeye/TinyLLama-v0

gpt-neo-125m: EleutherAI/gpt-neo-125m

opt-125m: facebook/opt-125m
```

Drop new entries in `model_registry.yaml`; aliases are auto-discovered.

## 9 · Performance vs. footprint (sample, SWORDS dev, M2 Pro CPU)

| Model         | RAM GB | P@1  | R@5  | Jaccard | Notes                 |
|---------------|--------|------|------|---------|-----------------------|
| **tinyllama** | 0.8    | 0.20 | 0.04 | 0.04    | Q4 GGUF, fast + stable|
| **llama3-mini**| 1.2   | 0.00 | 0.16 | 0.13    | Gated HF model        |
| **distilgpt2**| 1.1    | 0.10 | 0.05 | 0.08    | Compact transformer   |

> 🔧 *To improve performance, ensure `safetensors` is installed and configure `offload_folder` when using `accelerate`.*

## 10 · Citing

```bibtex
@software{lexsublm_lite_2025,
  author  = {Shamsuddin Ahmed},
  title   = {LexSubLM-Lite: Lightweight Contextual Lexical Substitution Toolkit},
  year    = {2025},
  url     = {https://github.com/shamspias/lexsublm-lite},
  license = {MIT}
}
```

## 11 · Licenses

* **Code** – MIT
* **Models** – See individual model cards (Apache-2, commercial, etc.)
* **Datasets** – CC-BY-4.0 unless noted

## 12 · Roadmap

* 🔜 LoRA fine-tuning on SWORDS (opt-in GPU)
* 🔜 Gradio playground demo
* 🔜 Multilingual eval on TSAR-2022 ES/PT

## 13 · 🐛 Known Issues

<details>
<summary><strong>🔐 Hugging Face: Gated model access (e.g., LLaMA 3)</strong></summary>

If you encounter:

```
401 Client Error: Unauthorized for url: https://huggingface.co/.../config.json
```

You need to authenticate for gated models:

1. **Login via CLI:**
   ```bash
   huggingface-cli login
   ```
2. **Obtain a token** from https://huggingface.co/settings/tokens
3. **Export** it:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```
4. **(Optional)** Request access: https://huggingface.co/meta-llama/Llama-3.2-1B

</details>

