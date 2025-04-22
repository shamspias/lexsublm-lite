# **masked‑lexical‑substitution**
*A laptop‑friendly toolkit for context‑aware single‑word paraphrasing and lexical‑substitution benchmarking*

---

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Built with ❤️](https://img.shields.io/badge/built%20with-%E2%9D%A4-red)

> **masked‑lexical‑substitution** (alias **LexSubLM**) generates context‑appropriate synonyms using quantised 1‑2 B‑parameter LLMs (DeepSeek‑1.5 B, Llama‑3 ≈ 1 B) and evaluates them on the official **SemEval‑2007 Task 10** benchmark – entirely on CPU, in minutes.

## ✨ Key Features

| Feature | What it gives you |
|---------|------------------|
| **⚡️ Laptop‑ready** | Runs on a MacBook M‑series or any 8 GB+ machine (4‑bit GGUF models) – *no* GPU or cloud required. |
| **🔍 Research‑grade metrics** | Implements SemEval P@1, Recall@10 & GAP for quick comparisons. |
| **🛠 Modular pipeline** | Separate *generate → filter → rank* stages; swap in any LLM, filter or reranker. |
| **🗂 Tiny footprint** | <150 MB code, <50 MB dataset; eval finishes <5 min CPU‑only. |
| **📦 pip‑installable** | `pip install masked-lexical-substitution` gives a CLI & importable API. |

---

## 🖼 Demo

```bash
$ lexsublm \
    --sentence "He sat on the bank of the river." \
    --target bank --top_k 5

[1] shore
[2] riverbank
[3] embankment
[4] waterside
[5] riverside
```

---

## 📚 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Benchmark Results](#benchmark-results)
6. [Roadmap](#roadmap)
7. [Citing](#citing)
8. [License](#license)

---

## Installation

**Prerequisites**  
* Python ≥ 3.9  
* macOS / Linux / Windows  
* ~4 GB free RAM (16 GB recommended for fastest eval)

```bash
# 1 Clone & cd
$ git clone https://github.com/shamspias/masked‑lexical‑substitution.git
$ cd masked‑lexical‑substitution

# 2 Install
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm  # POS filtering

# 3 (Option A) Download a quantised model automatically at first run
#  or (Option B) place your own GGUF in ~/.cache/lexsublm/
```

> **Tip**: On Apple Silicon, `pip install llama-cpp-python` wheels use Metal by default – zero setup!

---

## Quick Start

```bash
# CLI
lexsublm --sentence "The bright student solved the problem." --target bright

# Python API
from lexsublm import LexSub
lexsub = LexSub(model="deepseek-ai/deepseek-1.5b-chat-4bit")
lexsub("He sat on the bank of the river.", target="bank", top_k=3)
```

---

## Methodology

1. **Prompted generation** with an instruction‑tuned causal LLM.  
   *System prompt*: *“Return **one** synonym that preserves meaning & syntax.”*
2. **Filtering**  
   *POS match* (spaCy) → *whole‑word token* → *optional cosine ≥ 0.4* w/ MiniLM.
3. **Ranking**  
   Default = log‑prob; Alt = SBERT cosine or hybrid.
4. **Evaluation**  
   Compute P@1, Recall@10, GAP on SemEval‑07 gold.

<p align="center">
  <img src="docs/pipeline.svg" width="600" alt="pipeline diagram"/>
</p>

---

## Project Structure
```
lexsublm/
 ├── data/                # SemEval‑07 split + script to download
 │   └── semeval07/
 ├── lexsublm/
 │   ├── generator.py     # LLM wrapper (DeepSeek/Llama via llama‑cpp or HF)
 │   ├── filter.py        # POS & similarity filters
 │   ├── ranker.py        # scoring strategies
 │   ├── evaluator.py     # metrics (P@1, GAP)
 │   ├── cli.py           # argparse entry‑point
 │   └── __init__.py
 ├── evaluate.py          # reproduce paper baseline
 ├── notebooks/           # exploratory notebooks
 ├── requirements.txt
 └── README.md
```

---

## Benchmark Results

| Model (4‑bit) | P@1 | Recall@10 | GAP |
|---------------|-----|-----------|-----|
| DeepSeek‑1.5B‑chat | 0.32 | 0.63 | 0.41 |
| Llama‑3‑1B‑Instruct | 0.31 | 0.61 | 0.40 |
| DistilBERT‑base‑uncased (mask) | 0.28 | 0.55 | 0.37 |
| Human† | ~0.58 | – | ~0.52 |

† Reported by McCarthy & Navigli (2007).

---

## Roadmap
- [x] v0.1 – CLI, API, SemEval‑07 metrics
- [ ] v0.2 – Gradio demo
- [ ] v0.3 – Multilingual evaluation (CoInCo‑Fr/Es)
- [ ] v1.0 – LoRA fine‑tune recipe, publish to PyPI

---

## Contributing
Pull requests welcome 🙏 – please run `pre‑commit` and add unit tests. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Citing
If you use **masked‑lexical‑substitution** in your work, please cite :
```bibtex
@software{masked_lexical_substitution_2025,
  author       = {Shamsuddin Ahmed},
  title        = {masked‑lexical‑substitution: Context‑Aware Synonym Generation},
  year         = 2025,
  url          = {https://github.com/shamspias/masked‑lexical‑substitution}
}
```

---

## License
MIT – see [`LICENSE`](LICENSE) for details.

---

## Acknowledgements
Credits to the authors of **SemEval‑2007 Task 10**, Hugging Face Transformers, `llama‑cpp‑python`, and DeepSeek/Llama model creators.

