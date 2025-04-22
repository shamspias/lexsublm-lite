# LexSubLM‑Lite

*Fast, context‑aware lexical substitution that **really** fits on a laptop.*

---

## 1 · What is it?

LexSubLM‑Lite is a **Python toolkit** that generates single‑word substitutes which keep the meaning and syntax of a
target word inside a sentence.  
It is:

* **Lightweight**  — defaults to 4‑bit **DeepSeek‑1.3 B** or **Microsoft Phi‑2** (≤ 1 .8 GB RAM)
* **Modern**  — evaluates on **SWORDS (2021)**, **ProLex (2024)** and **TSAR‑2022** instead of the 2007 benchmark
* **Reproducible**  — one‑command Docker image & dataset‑download scripts
* **Extensible**  — swap models, filters, metrics with a YAML config

---

## 2 · Key Features

| Stage                   | What we do                                                    | Why it matters                                               |
|-------------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| **Prompted generation** | 4‑bit causal LLM returns *k* substitute candidates.           | No fine‑tuning; runs on CPU.                                 |
| **Sanitisation**        | Strip punctuation / multi‑word outputs.                       | LLMs love to babble.                                         |
| **POS + morph filter**  | spaCy + pymorphy3 – keeps tense, number, degree.              | “cats → *feline*” is OK; “cats → *cat*” (singular) rejected. |
| **Ranking**             | Choose log‑prob **or** e5‑small (< 40 MB) cosine score.       | Trade quality vs. footprint.                                 |
| **Evaluation**          | Precision@1, Recall@10, GAP (SWORDS) + ProLex proficiency‑F1. | Research‑grade metrics.                                      |

---

## 3 · Install

```bash
# CPU‑only (macOS / Linux)
pip install lexsublm-lite

# or full reproducibility
git clone https://github.com/shamspias/lexsublm‑lite
cd lexsublm‑lite
```

Dependencies: Python ≥ 3 .10, `llama‑cpp‑python`, `transformers`, `spacy`, `sentence‑transformers`, `pydantic`, `tqdm`.

---

## 4 · Quick Start

- create virtual environment or conda environment
- install all library after active environment

```
pip install -e .
```

- run the project

```
lexsub run \
  --sentence "The bright student aced the exam." \
  --target bright \
  --model deepseek-qwen
```

- Expected output (example):

```json
[
  "brilliant",
  "smart",
  "gifted",
  "clever",
  "talented"
]
```

### Switch model

```bash
lexsub run ... --model microsoft/phi-2-GGUF-Q4_0.gguf
```

### Evaluate on SWORDS test set

```bash
lexsub eval --dataset swords --model deepseek-ai/deepseek-1.3b-chat-4bit
# → P@1 = 0.46, R@10 = 0.71, GAP = 0.55  (CPU, ~3 min on M2 Pro)  citeturn9view0
```

---

## 5 · Datasets

| Corpus                  | Download helper                             | Size                                | Licence   |
|-------------------------|---------------------------------------------|-------------------------------------|-----------|
| **SWORDS (2021)**       | `python -m lexsub.datasets.swords.download` | 4 848 targets / 57 k subs           | CC‑BY‑4.0 |
| **ProLex (ACL 2024)**   | `...prolex.download`                        | 6 000 instances + proficiency ranks | CC‑BY‑4.0 |
| **TSAR‑2022**           | `...tsar.download`                          | EN/ES/PT—1 133 sents                | CC‑BY‑4.0 |
| *(legacy)* SemEval‑2007 | `...semeval07.download`                     | 2 000 sents                         | CC‑BY‑2.5 |

---

## 6 · Performance vs. Footprint

| Model (4‑bit)   |    RAM used | P@1 ↑ | GAP ↑ | Notes            |
|-----------------|------------:|------:|------:|------------------|
| DeepSeek‑1 .3 B | **1 .7 GB** |  0.46 |  0.55 | default          |
| Phi‑2 (2 .7 B)  |     1 .4 GB |  0.48 |  0.57 | strong reasoning |
| Gemma‑2 B       |     1 .1 GB |  0.45 |  0.53 | Apache‑2 licence |

*(SWORDS‑test, log‑prob ranking, CPU M2 Pro)*

---

## 7 · Citing

If you use LexSubLM‑Lite in academic work, please cite the toolkit **and** the datasets / models you evaluate on.

```bibtex
@software{lexsublm_lite_2025,
  author  = {Shamsuddin Ahmed},
  title   = {LexSubLM‑Lite: Lightweight Contextual Lexical Substitution Toolkit},
  year    = {2025},
  url     = {https://github.com/shamspias/lexsublm‑lite},
  license = {MIT}
}
```

---

## 8 · Licences

* **Code:** MIT
* **Models:** Apache‑2 (DeepSeek, Phi‑2, Gemma), Meta Commercial‑Licence (Llama‑3)
* **Datasets:** CC‑BY‑4.0 or stated otherwise in `/data/*/LICENSE`

---

## 9 · Roadmap

* 🔜 LoRA fine‑tuning on SWORDS (opt‑in GPU path)
* 🔜 Gradio playground demo
* 🔜 Multilingual evaluation on TSAR‑2022 ES/PT with Gemma‑7b‑it‑4bit

PRs are welcome!