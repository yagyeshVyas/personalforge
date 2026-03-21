<div align="center">

# PersonalForge v10

### Fine-tune any LLM on your own data — free, offline, no coding

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Cost: $0.00](https://img.shields.io/badge/Cost-%240.00_Forever-brightgreen.svg)]()
[![No Coding](https://img.shields.io/badge/Coding-Zero_Required-orange.svg)]()
[![Offline](https://img.shields.io/badge/Runs-100%25_Offline-purple.svg)]()

**Upload files → Stream datasets → Clean data → Pick any model → Train on free Colab → Download GGUF → Chat offline forever**

*No cloud. No subscriptions. No data sent anywhere. Everything stays on your machine.*

[GitHub](https://github.com/yagyeshVyas/personalforge) · [LinkedIn](https://www.linkedin.com/in/yagyeshvyas/) · [LM Studio](https://lmstudio.ai) · [Ollama](https://ollama.com)

</div>

---

## What is PersonalForge?

Most "chat with your docs" tools search your files at runtime using RAG. PersonalForge goes deeper — it **bakes your knowledge directly into a model's weights** through fine-tuning. The GGUF file that comes out *is* your knowledge. No internet needed ever again.

```
Your Data (files, HuggingFace datasets, web search, Google Drive...)
                        ↓
           17-technique data cleaning pipeline
                        ↓
        Auto-generate Q&A pairs with thinking chains
                        ↓
     Fine-tune on FREE Google Colab T4 GPU
     SFT → DPO → BGE-M3 RAG → Auto evaluation
                        ↓
        Download GGUF (Q4_K_M, ~2-4GB)
                        ↓
        Chat offline forever — LM Studio or Ollama
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yagyeshVyas/personalforge
cd personalforge

# 2. Install
pip install -r requirements.txt

# 3. Run
python run.py
```

Browser opens at **http://localhost:5000**

---

## Features

### Data Sources — 6 ways to add training data

| Source | What it does |
|--------|-------------|
| **File Upload** | PDF, Word, Excel, CSV, TXT, Python, JS, Java, C++, Jupyter |
| **HuggingFace Streaming** | 26 pre-verified free datasets, 1M-2M samples, no full download |
| **Web Search** | Wikipedia, arXiv, Stack Overflow, GitHub READMEs, Project Gutenberg |
| **URLs** | Any webpage, Wikipedia article, arXiv paper, GitHub repo |
| **Cloud & Remote** | Google Drive, Dropbox, AWS S3, GitHub Gist, Pastebin, JSON APIs |
| **Manual HF Dataset** | Paste any `owner/dataset-name` from HuggingFace |

### HuggingFace Dataset Registry — 26 verified datasets

| Category | Top Datasets |
|----------|-------------|
| Coding | `bigcode/starcoderdata`, `nampdn-ai/tiny-codes`, `sahil2801/CodeAlpaca-20k` |
| Chat | `HuggingFaceH4/ultrachat_200k`, `Open-Orca/OpenOrca`, `tatsu-lab/alpaca` |
| Reasoning | `nvidia/OpenMathInstruct-2`, `microsoft/orca-math-word-problems-200k` |
| General | `HuggingFaceFW/fineweb`, `wikimedia/wikipedia` (10 languages) |
| Medical | `medalpaca/medical_meadow_medqa`, `medalpaca/medical_meadow_wikidoc` |
| Legal | `pile-of-law/pile-of-law` |
| Science | `allenai/peS2o`, `ccdv/arxiv-summarization` |
| Finance | `FinGPT/fingpt-sentiment-train`, `gbharti/finance-alpaca` |
| Multilingual | `CohereForAI/aya_dataset` (65 languages) |

### Universal Model Support

Search or paste **any HuggingFace model ID** — auto-detects:

- Architecture (Llama, Qwen, Mistral, Phi, Gemma, DeepSeek, Falcon...)
- LoRA target modules
- Chat template format
- Optimal training config (LoRA rank, learning rate, batch size)

**58 popular models pre-configured** — or paste any model ID directly:
```
unsloth/Qwen3.5-4B
unsloth/Qwen2.5-Coder-7B-Instruct
unsloth/Llama-3.2-3B-Instruct
unsloth/DeepSeek-R1-Distill-Qwen-7B
google/gemma-2-9b-it
microsoft/phi-4
... and any other public HuggingFace model
```

### 17-Technique Data Cleaning Pipeline

| # | Technique | What it removes |
|---|-----------|----------------|
| 1 | Encoding fix | PDF conversion artifacts (`â€™` → `'`) |
| 2 | Unicode normalize | Zero-width chars, special spaces |
| 3 | PII removal | HF tokens, AWS keys, OpenAI keys, GitHub tokens |
| 4 | Page numbers | `Page 42`, `- 42 -`, standalone numbers |
| 5 | Headers/footers | ISBN, "Table of Contents", "Printed in..." |
| 6 | Watermarks | CONFIDENTIAL, DRAFT, DO NOT DISTRIBUTE |
| 7 | Boilerplate | Copyright lines, "All rights reserved" |
| 8 | Broken sentences | PDF line breaks (`sen-\ntence` → `sentence`) |
| 9 | Table noise | `|---|---|---` border lines |
| 10 | Whitespace | 3+ newlines, trailing spaces, tabs |
| 11 | Repeated chars | `==========`, `----------` separators |
| 12 | Code cleaning | Auto-generated markers, minified lines |
| 13 | URL removal | Raw URLs in non-code text |
| 14 | Quality gate | Rejects chunks under 80 chars or 15 words |
| 15 | Exact dedup | MD5 hash deduplication |
| 16 | Near dedup | 3-gram MinHash — removes 80%+ similar chunks |
| 17 | Sentence dedup | Cross-document sentence deduplication |

### 3 Training Modes

| Mode | Best For | How It Thinks |
|------|---------|--------------|
| **Developer / Coder** | Code, technical docs | Step-by-step solutions, code examples, best practices |
| **Deep Thinker** | Research, analysis | Multi-angle reasoning, connects ideas across documents |
| **Honest / Factual** | Manuals, reference | Cites sources, admits gaps, never guesses |

Every mode adds `<think>...</think>` reasoning chains before every answer.

---

## Fine-Tuning Pipeline

### Advanced techniques used in every training run

| Technique | Details |
|-----------|---------|
| **rsLoRA** | Rank-stabilized LoRA — more stable than standard LoRA |
| **NEFTune** | Noise embedding fine-tuning (α=5-10) — better generalization |
| **Gradient checkpointing** | 60% less VRAM via Unsloth |
| **8-bit AdamW** | Quantized optimizer — less VRAM |
| **Cosine LR + warmup** | Better convergence than linear schedule |
| **Gradient clipping** | Prevents loss spikes |
| **Early stopping** | Auto-loads best checkpoint |
| **DPO alignment** | Teaches preferred vs rejected responses |
| **Curriculum learning** | Easy → hard sorting |
| **Mixed precision** | BF16 on Ampere, FP16 on older GPUs |
| **4-bit quantization** | Fits 7B on free T4 GPU |
| **BGE-M3 RAG** | Best free embeddings + ChromaDB vector index |
| **Auto evaluation** | Accuracy score before download |

### LoRA Config by Model Size

```
Small (1-2B):  r=16, alpha=16, lr=3e-4, batch=4×2=8
Medium (3-4B): r=16, alpha=32, lr=2e-4, batch=2×4=8
Large (7B+):   r=8,  alpha=16, lr=1e-4, batch=1×8=8
```

### For best small model performance (3-4B punching above weight)

```python
neftune_noise_alpha = 10   # Higher than default 5
r                   = 32   # Double rank for small models
lora_alpha          = 64   # 2x rank
```

---

## Sample Targets

| Size | Samples | Quality | Training time |
|------|---------|---------|--------------|
| Quick test | 10K | Basic | ~15 min |
| Small | 50K | Decent | ~25 min |
| **Recommended** | **200K** | **Good** | **~45 min** |
| Large | 500K | Great | ~70 min |
| Production | 1M | Excellent | ~90 min* |
| Massive | 2M | Best | ~90 min* |

*Colab T4 caps at ~100K pairs per session. For 1M+ samples, generate in batches.

---

## Supported Models

### Best model per use case

| Use Case | Recommended | Why |
|----------|------------|-----|
| **Coding** | `unsloth/Qwen2.5-Coder-7B-Instruct` | Purpose-built for code |
| **Coding (small)** | `unsloth/Qwen3.5-4B` | Best under 4B, built-in thinking |
| **Reasoning** | `unsloth/DeepSeek-R1-Distill-Qwen-7B` | Built-in chain-of-thought |
| **Chat** | `unsloth/Llama-3.2-3B-Instruct` | Most natural conversations |
| **General** | `unsloth/Qwen2.5-7B-Instruct` | Best all-around |
| **Medical** | `unsloth/Phi-3-mini-4k-instruct` | Best factual accuracy |
| **Legal** | `unsloth/mistral-7b-instruct-v0.3` | Best for long documents |
| **Multilingual** | `unsloth/Qwen2.5-3B-Instruct` | Strongest multilingual |

---

## Running on Google Colab (Free)

After generating your notebook in PersonalForge:

```
1. File → Upload notebook → build_my_ai.ipynb
2. Runtime → Change runtime type → T4 GPU → Save  ⚠️ DO THIS FIRST
3. Run Cell 1 → upload training_pairs.jsonl when asked
4. Runtime → Run All
5. Watch eval_loss go down — auto-stops at best checkpoint
6. Last cell downloads your .gguf automatically
```

**Understanding loss numbers:**
```
eval_loss > 2.0  →  just started
eval_loss ~ 1.5  →  learning
eval_loss ~ 1.0  →  good model
eval_loss < 0.7  →  excellent model
eval_loss rising →  overfitting — training correctly auto-stopped
```

---

## Using Your GGUF

**LM Studio (easiest)**
```
Download from lmstudio.ai
File → Load Model → select .gguf → Chat
```

**Ollama**
```bash
ollama run /path/to/your-model.gguf
```

**Share on HuggingFace (optional)**
```
huggingface.co/new → upload .gguf
```

---

## Project Structure

```
personalforge/
├── server.py                    # Flask web server
├── run.py                       # Launch script
├── requirements.txt
├── templates/
│   └── index.html               # Full UI (single page app)
├── core/
│   ├── hw_scanner.py            # Hardware detection (RAM/GPU/disk)
│   ├── file_loader.py           # Multi-format file loader
│   ├── data_cleaner.py          # 17-technique cleaning pipeline
│   ├── hf_streamer.py           # HuggingFace dataset streaming
│   ├── hf_registry.py           # 26 verified dataset configs
│   ├── url_fetcher.py           # URL/webpage fetcher
│   ├── remote_fetcher.py        # Google Drive/Dropbox/S3
│   ├── web_collector.py         # Web search data collector
│   ├── pair_generator.py        # Q&A pair generation (3 modes)
│   ├── model_resolver.py        # Universal model config resolver
│   └── model_matcher.py         # Hardware-aware model matching
├── colab/
│   └── notebook_generator.py    # Colab notebook generator
├── data/                        # Uploaded files (gitignored)
└── output/                      # Generated files (gitignored)
    ├── training_pairs.jsonl
    └── build_my_ai.ipynb
```

---

## Free Stack

| Tool | Purpose |
|------|---------|
| [Unsloth](https://github.com/unslothai/unsloth) | 2x faster training, 60% less VRAM |
| [Google Colab](https://colab.research.google.com) | Free T4 GPU (~12 hrs/day) |
| [LM Studio](https://lmstudio.ai) | Run GGUF locally (desktop) |
| [Ollama](https://ollama.com) | Run GGUF locally (terminal) |
| [ChromaDB](https://www.trychroma.com) | Local vector database for RAG |
| [Flask](https://flask.palletsprojects.com) | Web UI framework |
| DuckDuckGo API | Free web search, no API key |
| Project Gutenberg | Free books API |

**Total cost: $0.00 — forever.**

---

## Honest Limitations

- Fine-tuning is **not** training from scratch — base model weights are preserved
- You need **200K+ pairs** for good results — fewer gives weaker output
- Free Colab T4 handles **~100K pairs per session** comfortably
- **7B models** are tight on free T4 (14GB VRAM needed) — use 4-bit
- RAG vs fine-tuned knowledge **conflict** is not fully resolved
- Web collection is **slower** than HF dataset streaming for large volumes

---

## Author

**Yagyesh Vyas**

- GitHub: [@yagyeshVyas](https://github.com/yagyeshVyas)
- LinkedIn: [yagyeshvyas](https://www.linkedin.com/in/yagyeshvyas/)
- Project: [github.com/yagyeshVyas/personalforge](https://github.com/yagyeshVyas/personalforge)

---

## License

MIT — free forever. Use it, modify it, share it.

---

<div align="center">

**Built with care · Free forever · MIT License**

[⭐ Star this repo](https://github.com/yagyeshVyas/personalforge) · [🐛 Report Bug](https://github.com/yagyeshVyas/personalforge/issues) · [💡 Request Feature](https://github.com/yagyeshVyas/personalforge/issues)

*Powered by [Unsloth](https://github.com/unslothai/unsloth) · [Google Colab](https://colab.research.google.com) · [llama.cpp](https://github.com/ggerganov/llama.cpp)*

</div>