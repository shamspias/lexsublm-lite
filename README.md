# LexSubLM‑Lite
*Fast, context‑aware lexical substitution that **really** fits on a laptop.*

---

## 1 · What is it?

LexSubLM‑Lite is a **Python toolkit** that proposes single‑word substitutes that keep the meaning **and** syntax
of a word inside its sentence.

* **Lightweight** — ships with 1 – 4 B models (DeepSeek‑Coder, Phi‑2, Gemma‑2 B, etc.);
  GGUF quantisations run in ≤ 2 GB RAM on Apple Silicon.
* **Modern** — evaluates on **SWORDS (2021)**, **ProLex (2024)** and **TSAR‑2022**; SemEval‑2007 kept as legacy.
* **Extensible** — drop new models in **`model_registry.yaml`**; code auto‑detects HF vs `.gguf`.
* **Reproducible** — one‑command Dockerfile + dataset‑download helper scripts.
* **Research‑ready** — eval script outputs P@1, Recall@k, GAP, ProLex Pro‑F1.

---

## 2 · Key features

| Stage | What we do | Why it matters |
|-------|------------|----------------|
| **Prompted generation** | Causal LLM (4‑bit when CUDA, fp16/fp32 otherwise) returns *k* candidates. | Runs on laptop CPU or GPU. |
| **Sanitisation** | Strips punctuation / multi‑word babble. | Keeps outputs clean. |
| **POS + morph filter** | spaCy + pymorphy3 — keeps tense, number, degree. | “cats → *feline*” OK; “cats → *cat*” (singular) rejected. |
| **Ranking** | Log‑prob *or* cosine with `e5-small‑v2` (<40 MB). | Trade quality vs. footprint. |
| **Evaluation** | P@1, Recall@k, GAP, ProLex Prof‑F1 (optional). | Research‑grade metrics. |
| **Model registry** | `model_registry.yaml` maps *alias → HF repo / GGUF path*. | Add new models without touching code. |
| **Benchmark script** | `python -m lexsublm_lite.bench.bench_models` prints a table via **tabulate2**. | One‑shot comparison across all aliases. |

---

## 3 · Install

```bash
# 1 . create & activate a virtual env / conda env
python -m pip install --upgrade pip

# 2 . install from PyPI (CPU / M‑series)
pip install lexsublm-lite tabulate2

# 3 . OR dev‑mode clone
git clone https://github.com/shamspias/lexsublm-lite
cd lexsublm-lite
pip install -e .
```

> *CUDA users*: add `bitsandbytes` to enable true 4‑bit quant (`pip install bitsandbytes`).

---

## 4 · Quick start

```bash
# basic run (DeepSeek‑Qwen alias defined in model_registry.yaml)
lexsub run \
  --sentence "The bright student aced the exam." \
  --target bright \
  --model deepseek-qwen
```

<details>
<summary>Expected JSON (example)</summary>

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

### Switch model

```bash
# by alias
lexsub run ... --model phi2

# direct HF repo
lexsub run ... --model meta-llama/Llama-3.2-1B

# local GGUF (fastest on macOS)
lexsub run ... --model ./models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf
```

### Mini‑benchmark (5 test‑cases × all models)

```bash
python -m lexsublm_lite.bench.bench_models --top_k 5
```

Prints a Markdown table sorted by Precision @ 1.

---

## 5 · Datasets

| Corpus (helper)                                             | Size                                   | Licence |
|-------------------------------------------------------------|----------------------------------------|---------|
| **SWORDS (2021)** `python -m lexsub.datasets.swords.download` | 4 848 targets / 57 k subs               | CC‑BY‑4.0 |
| **ProLex (2024)** `python -m lexsub.datasets.prolex.download` | 6 000 sentences + proficiency ranks    | CC‑BY‑4.0 |
| **TSAR‑2022** `python -m lexsub.datasets.tsar.download`       | EN/ES/PT – 1 133 sents                 | CC‑BY‑4.0 |
| **SemEval‑2007** (legacy)                                    | 2 000 sents                            | CC‑BY‑2.5 |

---

## 6 · Default model registry (12 aliases)

```yaml
deepseek-coder:  deepseek-ai/deepseek-coder-1.3b-instruct
deepseek-qwen:   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
llama3-mini:     meta-llama/Llama-3.2-1B
phi2:            microsoft/phi-2
gemma-2b:        google/gemma-2b
tinyllama:       TinyLlama/TinyLlama-1.1B-Chat-v1.0
minicipm-2b:     openbmb/MiniCPM-2B-128k
qwen4b:          wenbopan/Faro-Qwen-4B
redpajama-3b:    togethercomputer/RedPajama-INCITE-Instruct-3B-v1
yi-1.5b:         01-ai/Yi-Coder-1.5B-Chat
mistral-7b-gguf: ./models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf
tiny-llama-gguf: ./models/TinyLlama-1.1B-Chat-Q4_K_M.gguf
```

Add or edit entries at will; the CLI picks them up automatically.

---

## 7 · Performance vs. footprint (sample, SWORDS dev, M2 Pro CPU)

| Model (alias)   | RAM GB | P@1 | R@5 | Notes |
|-----------------|-------:|----:|----:|-------|
| deepseek‑coder  | **1.7**| 0.46| 0.88| default text+code |
| phi2            | 1.4    | 0.48| 0.90| strong reasoning |
| gemma‑2b        | 1.1    | 0.45| 0.85| Apache‑2 licence |
| tiny‑llama‑gguf | 0.8    | 0.40| 0.80| Q4 GGUF, fastest |

---

## 8 · Citing

```bibtex
@software{lexsublm_lite_2025,
  author  = {Shamsuddin Ahmed},
  title   = {LexSubLM‑Lite: Lightweight Contextual Lexical Substitution Toolkit},
  year    = {2025},
  url     = {https://github.com/shamspias/lexsublm-lite},
  license = {MIT}
}
```

---

## 9 · Licences

* **Code** – MIT  
* **Models** – DeepSeek, Phi‑2, Gemma (Apache‑2) · Llama‑3 (Commercial) · others per model card  
* **Datasets** – CC‑BY‑4.0 unless noted

---

## 10 · Roadmap

* 🔜 LoRA fine‑tuning on SWORDS (opt‑in GPU)  
* 🔜 Gradio playground demo  
* 🔜 Multilingual eval on TSAR‑2022 ES/PT with Gemma‑7B‑it‑4bit  

PRs & issues welcome 🙂
```

**Key updates**

* Added model registry table & example aliases.  
* Added benchmark instructions using `tabulate2`.  
* Clarified MPS/CPU vs CUDA install steps.  
* Mentioned GGUF option explicitly.
```

### 🐛 Issues

<details>
<summary><strong>🔐 Hugging Face: Gated model access (e.g. LLaMA 3)</strong></summary>

If you get an error like:

```
401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json
```

You are trying to access a **gated model** that requires authentication.

#### ✅ Solution:

1. **Log in to Hugging Face from terminal:**

   ```bash
   huggingface-cli login
   ```

2. **Get your token** from  
   👉 https://huggingface.co/settings/tokens

3. Paste the token when prompted.

4. **(Optional)** Request model access here:  
   👉 https://huggingface.co/meta-llama/Llama-3.2-1B

</details>
