# core/model_matcher.py — Auto-match best model based on hardware + use case
# Scans device RAM/GPU first, then recommends from HuggingFace
import requests, logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ─── USE CASE CATEGORIES ──────────────────────────────────────────────────────

CATEGORIES = {
    "coding": {
        "name":        "Coding / Development",
        "icon":        "💻",
        "description": "Write code, debug, explain programming concepts",
        "keywords":    ["python","javascript","coding","programming","developer","software"],
        "models": [
            {
                "name":        "Qwen2.5-Coder 1.5B",
                "hf_id":       "unsloth/Qwen2.5-Coder-1.5B-Instruct",
                "params":      "1.5B",
                "ram_needed":  4,
                "vram_needed": 5,
                "gguf_gb":     1.1,
                "colab_min":   20,
                "why":         "Best coding model at 1.5B. Trained specifically on code.",
                "recommended": True,
            },
            {
                "name":        "Qwen2.5-Coder 3B",
                "hf_id":       "unsloth/Qwen2.5-Coder-3B-Instruct",
                "params":      "3B",
                "ram_needed":  6,
                "vram_needed": 8,
                "gguf_gb":     2.0,
                "colab_min":   40,
                "why":         "Excellent code generation. Best for complex projects.",
                "recommended": False,
            },
            {
                "name":        "Qwen2.5-Coder 7B",
                "hf_id":       "unsloth/Qwen2.5-Coder-7B-Instruct",
                "params":      "7B",
                "ram_needed":  8,
                "vram_needed": 14,
                "gguf_gb":     4.5,
                "colab_min":   80,
                "why":         "Top-tier coding. Handles complex algorithms and large codebases.",
                "recommended": False,
            },
        ],
    },
    "talking": {
        "name":        "Talking / Conversation",
        "icon":        "💬",
        "description": "Natural conversations, customer support, chatbot",
        "keywords":    ["chat","conversation","talking","assistant","support","dialogue"],
        "models": [
            {
                "name":        "Llama 3.2 1B",
                "hf_id":       "unsloth/Llama-3.2-1B-Instruct",
                "params":      "1B",
                "ram_needed":  4,
                "vram_needed": 5,
                "gguf_gb":     0.8,
                "colab_min":   15,
                "why":         "Fast, conversational. Great for simple Q&A chatbots.",
                "recommended": False,
            },
            {
                "name":        "Llama 3.2 3B",
                "hf_id":       "unsloth/Llama-3.2-3B-Instruct",
                "params":      "3B",
                "ram_needed":  6,
                "vram_needed": 8,
                "gguf_gb":     2.0,
                "colab_min":   40,
                "why":         "Natural, fluent conversations. Best for customer support.",
                "recommended": True,
            },
            {
                "name":        "Mistral 7B",
                "hf_id":       "unsloth/mistral-7b-instruct-v0.3",
                "params":      "7B",
                "ram_needed":  8,
                "vram_needed": 14,
                "gguf_gb":     4.5,
                "colab_min":   80,
                "why":         "Most natural conversations. Best for complex dialogue.",
                "recommended": False,
            },
        ],
    },
    "reasoning": {
        "name":        "Reasoning / Analysis",
        "icon":        "🧠",
        "description": "Deep thinking, analysis, research, complex problem solving",
        "keywords":    ["research","analysis","reasoning","thinking","science","academic"],
        "models": [
            {
                "name":        "DeepSeek-R1 1.5B",
                "hf_id":       "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
                "params":      "1.5B",
                "ram_needed":  4,
                "vram_needed": 5,
                "gguf_gb":     1.1,
                "colab_min":   20,
                "why":         "Built-in reasoning chains. Thinks before answering.",
                "recommended": False,
            },
            {
                "name":        "DeepSeek-R1 7B",
                "hf_id":       "unsloth/DeepSeek-R1-Distill-Qwen-7B",
                "params":      "7B",
                "ram_needed":  8,
                "vram_needed": 14,
                "gguf_gb":     4.5,
                "colab_min":   85,
                "why":         "Best reasoning quality. Complex analysis and research.",
                "recommended": True,
            },
        ],
    },
    "documents": {
        "name":        "Documents / Knowledge Base",
        "icon":        "📚",
        "description": "Answer questions from PDFs, manuals, notes, books",
        "keywords":    ["document","pdf","book","manual","knowledge","notes","reference"],
        "models": [
            {
                "name":        "Qwen2.5 1.5B",
                "hf_id":       "unsloth/Qwen2.5-1.5B-Instruct",
                "params":      "1.5B",
                "ram_needed":  4,
                "vram_needed": 5,
                "gguf_gb":     1.1,
                "colab_min":   20,
                "why":         "Good for simple document Q&A. Fast and light.",
                "recommended": False,
            },
            {
                "name":        "Qwen2.5 3B",
                "hf_id":       "unsloth/Qwen2.5-3B-Instruct",
                "params":      "3B",
                "ram_needed":  6,
                "vram_needed": 8,
                "gguf_gb":     2.0,
                "colab_min":   40,
                "why":         "Great document understanding. Recommended for most cases.",
                "recommended": True,
            },
            {
                "name":        "Qwen2.5 7B",
                "hf_id":       "unsloth/Qwen2.5-7B-Instruct",
                "params":      "7B",
                "ram_needed":  8,
                "vram_needed": 14,
                "gguf_gb":     4.5,
                "colab_min":   80,
                "why":         "Best comprehension for complex documents.",
                "recommended": False,
            },
        ],
    },
    "medical": {
        "name":        "Medical / Healthcare",
        "icon":        "🏥",
        "description": "Medical notes, clinical docs, health Q&A",
        "keywords":    ["medical","health","clinical","doctor","patient","drug","diagnosis"],
        "models": [
            {
                "name":        "Phi-3 Mini 3.8B",
                "hf_id":       "unsloth/Phi-3-mini-4k-instruct",
                "params":      "3.8B",
                "ram_needed":  6,
                "vram_needed": 10,
                "gguf_gb":     2.4,
                "colab_min":   45,
                "why":         "Strong factual accuracy. Good for structured medical info.",
                "recommended": True,
            },
        ],
    },
    "legal": {
        "name":        "Legal / Law",
        "icon":        "⚖️",
        "description": "Legal documents, case analysis, law Q&A",
        "keywords":    ["legal","law","contract","court","attorney","case","regulation"],
        "models": [
            {
                "name":        "Mistral 7B",
                "hf_id":       "unsloth/mistral-7b-instruct-v0.3",
                "params":      "7B",
                "ram_needed":  8,
                "vram_needed": 14,
                "gguf_gb":     4.5,
                "colab_min":   80,
                "why":         "Best for long, complex legal text comprehension.",
                "recommended": True,
            },
        ],
    },
    "multilingual": {
        "name":        "Multilingual",
        "icon":        "🌍",
        "description": "Multiple languages, translation, non-English content",
        "keywords":    ["multilingual","translation","hindi","arabic","chinese","french","spanish"],
        "models": [
            {
                "name":        "Qwen2.5 3B",
                "hf_id":       "unsloth/Qwen2.5-3B-Instruct",
                "params":      "3B",
                "ram_needed":  6,
                "vram_needed": 8,
                "gguf_gb":     2.0,
                "colab_min":   40,
                "why":         "Qwen2.5 has strongest multilingual support of all free models.",
                "recommended": True,
            },
        ],
    },
    "general": {
        "name":        "General Purpose",
        "icon":        "✨",
        "description": "General assistant, mixed use, everything",
        "keywords":    ["general","assistant","everything","mixed","all"],
        "models": [
            {
                "name":        "Qwen2.5 3B",
                "hf_id":       "unsloth/Qwen2.5-3B-Instruct",
                "params":      "3B",
                "ram_needed":  6,
                "vram_needed": 8,
                "gguf_gb":     2.0,
                "colab_min":   40,
                "why":         "Best all-around model for mixed use cases.",
                "recommended": True,
            },
        ],
    },
}


class ModelMatcher:

    def match(self, system_info: Dict, category: str,
              data_description: str = "") -> Dict:
        """
        Given hardware info and desired category,
        return the best matching model with explanation.
        """
        ram_gb   = system_info.get("ram_gb", 8)
        gpu      = system_info.get("gpu", {})
        vram_gb  = gpu.get("vram_gb", 0) if gpu.get("available") else 0

        cat_info = CATEGORIES.get(category, CATEGORIES["general"])
        models   = cat_info["models"]

        # Filter by hardware
        compatible = []
        for m in models:
            # Check if fits in RAM (for local inference)
            fits_ram  = m["ram_needed"] <= ram_gb
            # Check Colab feasibility (T4 has 15GB)
            fits_colab = m["vram_needed"] <= 14
            m["fits_ram"]   = fits_ram
            m["fits_colab"] = fits_colab
            m["hardware_warning"] = None

            if not fits_ram:
                m["hardware_warning"] = f"Needs {m['ram_needed']}GB RAM — your device has {ram_gb}GB. Will work on Colab but may be slow locally."
            if not fits_colab:
                m["hardware_warning"] = f"May be tight on free Colab T4 (needs {m['vram_needed']}GB VRAM)"

            compatible.append(m)

        # Pick best match
        best = None
        for m in compatible:
            if m.get("recommended") and m["fits_ram"]:
                best = m
                break
        if not best:
            # Pick largest that fits RAM
            fitting = [m for m in compatible if m["fits_ram"]]
            if fitting:
                best = fitting[-1]
            else:
                best = compatible[0]  # recommend anyway with warning

        return {
            "category":   category,
            "cat_info":   cat_info,
            "all_models": compatible,
            "best":       best,
            "system":     system_info,
        }

    def auto_detect_category(self, file_types: Dict, keywords: str = "") -> str:
        """Auto-detect best category from uploaded file types and keywords."""
        kw = keywords.lower()

        # Check keywords first
        for cat, info in CATEGORIES.items():
            if any(k in kw for k in info["keywords"]):
                return cat

        # Check file types
        counts = file_types or {}
        code_count = counts.get("code", 0)
        doc_count  = counts.get("document", 0) + counts.get("notes", 0)
        data_count = counts.get("data", 0)

        if code_count > doc_count:
            return "coding"
        if doc_count > 0:
            return "documents"
        if data_count > 0:
            return "general"

        return "general"

    def get_all_categories(self) -> List[Dict]:
        return [
            {
                "id":          cat_id,
                "name":        info["name"],
                "icon":        info["icon"],
                "description": info["description"],
            }
            for cat_id, info in CATEGORIES.items()
        ]
