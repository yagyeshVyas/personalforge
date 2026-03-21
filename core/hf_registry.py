# core/hf_registry.py — VERIFIED dataset registry
# Every config/split/field verified against HuggingFace documentation
# No loading scripts — all Parquet-based

REGISTRY = {

    "coding": {
        "name": "Coding / Programming", "icon": "💻",
        "target_samples": 1000000,
        "datasets": [
            {
                "id":          "bigcode/starcoderdata",
                "name":        "StarCoderData ⭐",
                "desc":        "Real code in 80+ languages. Load via data_dir='python'. Used to train StarCoder2.",
                "size":        "200GB total",
                "config":      "python",
                "split":       "train",
                "fields":      {"text": "content"},
                "data_dir":    True,
                "max_recommended": 500000,
                "free": True, "gated": False, "star": True,
                "note": "Available configs: python, javascript, java, go, rust, cpp, typescript, php, ruby, c, scala, shell",
                "load_example": 'load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)',
            },
            {
                "id":          "sahil2801/CodeAlpaca-20k",
                "name":        "CodeAlpaca 20K ⭐",
                "desc":        "20K coding instruction pairs. config=default, split=train.",
                "size":        "20K samples",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "instruction", "output": "output"},
                "max_recommended": 20000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "iamtarun/python_code_instructions_18k_alpaca",
                "name":        "Python Instructions 18K",
                "desc":        "18K Python coding instructions. config=default, split=train.",
                "size":        "18K samples",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "instruction", "output": "output"},
                "max_recommended": 18000,
                "free": True, "gated": False, "star": False,
            },
            {
                "id":          "bigcode/bigcodebench",
                "name":        "BigCodeBench",
                "desc":        "Hard coding tasks with unit tests. split=v0.1.4.",
                "size":        "1.1K problems",
                "config":      None,
                "split":       "v0.1.4",
                "fields":      {"instruction": "instruct_prompt", "output": "canonical_solution"},
                "max_recommended": 1140,
                "free": True, "gated": False, "star": False,
            },
            {
                "id":          "nampdn-ai/tiny-codes",
                "name":        "Tiny Codes 974K",
                "desc":        "974K code generation pairs. config=default, split=train.",
                "size":        "974K samples",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "prompt", "output": "response"},
                "max_recommended": 500000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "Nan-Do/code-search-net-python",
                "name":        "CodeSearchNet Python",
                "desc":        "Clean Python code with docstrings. split=train.",
                "size":        "412K samples",
                "config":      None,
                "split":       "train",
                "fields":      {"text": "whole_func_string"},
                "max_recommended": 300000,
                "free": True, "gated": False, "star": False,
            },
            {
                "id":          "bigcode/the-stack-v2",
                "name":        "The Stack v2 🔒",
                "desc":        "67TB code in 600+ languages. Requires HF token + agreement.",
                "size":        "67TB",
                "config":      None,
                "split":       "train",
                "fields":      {"text": "content"},
                "max_recommended": 500000,
                "free": True, "gated": True,
                "token_url":   "https://huggingface.co/datasets/bigcode/the-stack-v2",
                "star": False,
            },
        ],
    },

    "chat": {
        "name": "Chat / Conversation", "icon": "💬",
        "target_samples": 500000,
        "datasets": [
            {
                "id":          "HuggingFaceH4/ultrachat_200k",
                "name":        "UltraChat 200K ⭐",
                "desc":        "200K multi-turn conversations. split=train_sft.",
                "size":        "200K",
                "config":      None,
                "split":       "train_sft",
                "fields":      {"messages": "messages"},
                "max_recommended": 200000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "Open-Orca/OpenOrca",
                "name":        "OpenOrca 1M ⭐",
                "desc":        "1M GPT-4 quality reasoning + chat. split=train.",
                "size":        "1M",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "question", "output": "response"},
                "max_recommended": 500000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "tatsu-lab/alpaca",
                "name":        "Stanford Alpaca 52K",
                "desc":        "52K instruction pairs. split=train.",
                "size":        "52K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "instruction", "output": "output"},
                "max_recommended": 52000,
                "free": True, "gated": False, "star": False,
            },
            {
                "id":          "databricks/databricks-dolly-15k",
                "name":        "Dolly 15K",
                "desc":        "15K human-written instruction pairs. High quality.",
                "size":        "15K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "instruction", "output": "response"},
                "max_recommended": 15000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "reasoning": {
        "name": "Reasoning / Math", "icon": "🧠",
        "target_samples": 1000000,
        "datasets": [
            {
                "id":          "nvidia/OpenMathInstruct-2",
                "name":        "OpenMathInstruct 2 ⭐",
                "desc":        "14M math instruction pairs. Best free math dataset.",
                "size":        "14M",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "problem", "output": "generated_solution"},
                "max_recommended": 1000000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "microsoft/orca-math-word-problems-200k",
                "name":        "Orca Math 200K ⭐",
                "desc":        "200K math word problems with reasoning steps.",
                "size":        "200K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "question", "output": "answer"},
                "max_recommended": 200000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "openai/gsm8k",
                "name":        "GSM8K",
                "desc":        "8.5K grade school math. config=main, split=train.",
                "size":        "8.5K",
                "config":      "main",
                "split":       "train",
                "fields":      {"instruction": "question", "output": "answer"},
                "max_recommended": 8500,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "general": {
        "name": "General Knowledge", "icon": "✨",
        "target_samples": 2000000,
        "datasets": [
            {
                "id":          "HuggingFaceFW/fineweb",
                "name":        "FineWeb 15T ⭐",
                "desc":        "15T tokens clean web text. config=default, split=train.",
                "size":        "15T tokens",
                "config":      None,
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 2000000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "wikimedia/wikipedia",
                "name":        "Wikipedia ⭐",
                "desc":        "Full Wikipedia. config=20231101.en (or other language).",
                "size":        "20GB/lang",
                "config":      "20231101.en",
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 500000,
                "free": True, "gated": False, "star": True,
                "configs_available": [
                    "20231101.en","20231101.hi","20231101.fr","20231101.de",
                    "20231101.es","20231101.zh","20231101.ar","20231101.ja",
                    "20231101.pt","20231101.ru"
                ],
            },
            {
                "id":          "allenai/dolma",
                "name":        "Dolma 3T",
                "desc":        "3T tokens web+books+code+papers. config=v1_6.",
                "size":        "3T tokens",
                "config":      "v1_6",
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 1000000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "medical": {
        "name": "Medical / Healthcare", "icon": "🏥",
        "target_samples": 200000,
        "datasets": [
            {
                "id":          "medalpaca/medical_meadow_medqa",
                "name":        "Medical MedQA ⭐",
                "desc":        "10K USMLE exam Q&A. split=train.",
                "size":        "10K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "input", "output": "output"},
                "max_recommended": 10000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "medalpaca/medical_meadow_wikidoc",
                "name":        "Medical WikiDoc",
                "desc":        "67K medical knowledge entries. split=train.",
                "size":        "67K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "input", "output": "output"},
                "max_recommended": 67000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "legal": {
        "name": "Legal / Law", "icon": "⚖️",
        "target_samples": 300000,
        "datasets": [
            {
                "id":          "pile-of-law/pile-of-law",
                "name":        "Pile of Law ⭐",
                "desc":        "Court opinions, contracts, regulations. config=courtlistener_opinions.",
                "size":        "256GB",
                "config":      "courtlistener_opinions",
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 300000,
                "free": True, "gated": False, "star": True,
            },
        ],
    },

    "science": {
        "name": "Science / Research", "icon": "🔬",
        "target_samples": 500000,
        "datasets": [
            {
                "id":          "allenai/peS2o",
                "name":        "peS2o Scientific Papers ⭐",
                "desc":        "Cleaned scientific papers. config=v2, split=train.",
                "size":        "40GB",
                "config":      "v2",
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 500000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "ccdv/arxiv-summarization",
                "name":        "ArXiv Summarization",
                "desc":        "215K ArXiv papers + abstracts. config=document.",
                "size":        "215K",
                "config":      "document",
                "split":       "train",
                "fields":      {"instruction": "article", "output": "abstract"},
                "max_recommended": 215000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "finance": {
        "name": "Finance / Business", "icon": "💹",
        "target_samples": 100000,
        "datasets": [
            {
                "id":          "FinGPT/fingpt-sentiment-train",
                "name":        "FinGPT Sentiment ⭐",
                "desc":        "76K financial news sentiment pairs. split=train.",
                "size":        "76K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "input", "output": "output"},
                "max_recommended": 76000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "gbharti/finance-alpaca",
                "name":        "Finance Alpaca",
                "desc":        "68K finance Q&A. split=train.",
                "size":        "68K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "instruction", "output": "output"},
                "max_recommended": 68000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },

    "multilingual": {
        "name": "Multilingual", "icon": "🌍",
        "target_samples": 500000,
        "datasets": [
            {
                "id":          "CohereForAI/aya_dataset",
                "name":        "Aya Dataset ⭐",
                "desc":        "204K instruction pairs in 65 languages. split=train.",
                "size":        "204K",
                "config":      None,
                "split":       "train",
                "fields":      {"instruction": "inputs", "output": "targets"},
                "max_recommended": 204000,
                "free": True, "gated": False, "star": True,
            },
            {
                "id":          "uonlp/CulturaX",
                "name":        "CulturaX 167 Languages",
                "desc":        "6.3T tokens in 167 languages. config=en.",
                "size":        "6.3T tokens",
                "config":      "en",
                "split":       "train",
                "fields":      {"text": "text"},
                "max_recommended": 500000,
                "free": True, "gated": False, "star": False,
            },
        ],
    },
}

BEST_MODELS = {
    "coding":       {"name":"Qwen2.5-Coder-7B",  "hf_id":"unsloth/Qwen2.5-Coder-7B-Instruct", "why":"Purpose-built for code."},
    "chat":         {"name":"Llama-3.2-3B",       "hf_id":"unsloth/Llama-3.2-3B-Instruct",     "why":"Best conversational model."},
    "reasoning":    {"name":"DeepSeek-R1-7B",     "hf_id":"unsloth/DeepSeek-R1-Distill-Qwen-7B","why":"Built-in chain-of-thought."},
    "general":      {"name":"Qwen2.5-7B",         "hf_id":"unsloth/Qwen2.5-7B-Instruct",       "why":"Best all-around model."},
    "medical":      {"name":"Phi-3-mini-3.8B",    "hf_id":"unsloth/Phi-3-mini-4k-instruct",    "why":"Best factual accuracy."},
    "legal":        {"name":"Mistral-7B",         "hf_id":"unsloth/mistral-7b-instruct-v0.3",  "why":"Best for long documents."},
    "science":      {"name":"Qwen2.5-7B",         "hf_id":"unsloth/Qwen2.5-7B-Instruct",       "why":"Strong scientific reasoning."},
    "finance":      {"name":"Qwen2.5-3B",         "hf_id":"unsloth/Qwen2.5-3B-Instruct",       "why":"Good balance."},
    "multilingual": {"name":"Qwen2.5-3B",         "hf_id":"unsloth/Qwen2.5-3B-Instruct",       "why":"Strongest multilingual."},
}

SAMPLE_GUIDE = {
    "minimal": {"n":10000,   "label":"10K — Quick test"},
    "small":   {"n":50000,   "label":"50K — Small"},
    "medium":  {"n":200000,  "label":"200K — Recommended"},
    "large":   {"n":500000,  "label":"500K — Large"},
    "xlarge":  {"n":1000000, "label":"1M — Production"},
    "massive": {"n":2000000, "label":"2M — Massive"},
}


def get_all():           return REGISTRY
def get_category(cat):   return REGISTRY.get(cat, {})
def get_best_model(cat): return BEST_MODELS.get(cat, BEST_MODELS["general"])
def get_sample_guide():  return SAMPLE_GUIDE

def get_dataset(ds_id):
    for cat in REGISTRY.values():
        for ds in cat.get("datasets", []):
            if ds["id"] == ds_id:
                return ds
    return {}

def search(q):
    q = q.lower()
    out = []
    for cat_id, cat in REGISTRY.items():
        for ds in cat.get("datasets", []):
            if q in ds["id"].lower() or q in ds["name"].lower() or q in ds["desc"].lower():
                out.append({**ds, "category": cat_id})
    return out

def all_dataset_ids():
    return [ds["id"] for cat in REGISTRY.values() for ds in cat.get("datasets", [])]
