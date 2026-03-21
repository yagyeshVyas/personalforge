# core/model_resolver.py — Universal HuggingFace Model Resolver
# Supports ANY model from HuggingFace
# Auto-detects: architecture, chat template, LoRA targets, optimal config
# Works with: Llama, Qwen, Mistral, Phi, Gemma, DeepSeek, Falcon, MPT, etc.

import re, logging, requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
HF_API = "https://huggingface.co/api"
HEADERS = {"User-Agent": "PersonalForge/10.0"}


# ── KNOWN ARCHITECTURE CONFIGS ────────────────────────────────────────────────
# For models not in Unsloth, we fall back to standard transformers

ARCH_CONFIGS = {
    # Llama family
    "llama": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "llama3",
        "eos":            "<|eot_id|>",
        "chat_format":    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant}<|eot_id|>",
    },
    "mistral": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "mistral",
        "eos":            "</s>",
        "chat_format":    "<s>[INST] {system}\n{user} [/INST]{assistant}</s>",
    },
    "qwen2": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "qwen",
        "eos":            "<|im_end|>",
        "chat_format":    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>",
    },
    "phi": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2"],
        "template":       "phi3",
        "eos":            "<|end|>",
        "chat_format":    "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>",
    },
    "gemma": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "gemma",
        "eos":            "<eos>",
        "chat_format":    "<bos><start_of_turn>user\n{system}\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>",
    },
    "falcon": {
        "target_modules": ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"],
        "template":       "falcon",
        "eos":            "<|endoftext|>",
        "chat_format":    "System: {system}\nUser: {user}\nAssistant: {assistant}",
    },
    "deepseek": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "deepseek",
        "eos":            "<｜end▁of▁sentence｜>",
        "chat_format":    "<｜begin▁of▁sentence｜>{system}\n\nUser: {user}\n\nAssistant: {assistant}<｜end▁of▁sentence｜>",
    },
    "gpt_neox": {
        "target_modules": ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"],
        "template":       "chatml",
        "eos":            "<|endoftext|>",
        "chat_format":    "<|system|>{system}<|user|>{user}<|assistant|>{assistant}",
    },
    "t5": {
        "target_modules": ["q","k","v","o","wi","wo"],
        "template":       "none",
        "eos":            "</s>",
        "chat_format":    "{system}\n\nQuestion: {user}\n\nAnswer: {assistant}",
    },
    "default": {
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "template":       "chatml",
        "eos":            "</s>",
        "chat_format":    "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}",
    },
}

# Unsloth supported models (faster training)
UNSLOTH_SUPPORTED = [
    "llama", "mistral", "qwen", "phi", "gemma", "deepseek",
    "tinyllama", "codellama", "wizardlm", "vicuna", "alpaca",
    "zephyr", "openchat", "starling", "neural-chat",
]

# ── BUILT-IN MODEL REGISTRY ──────────────────────────────────────────────────
# 80+ models — search works even without internet
# Format: id, name, cat, params, star, desc
POPULAR_MODELS = [
    # ── CODING ────────────────────────────────────────────────────────────────
    {"id":"unsloth/Qwen2.5-Coder-7B-Instruct",          "name":"Qwen2.5-Coder 7B",        "cat":"coding",    "params":"7B",   "star":True,  "desc":"Best free coding model. Top performance on code tasks."},
    {"id":"unsloth/Qwen2.5-Coder-3B-Instruct",          "name":"Qwen2.5-Coder 3B",        "cat":"coding",    "params":"3B",   "star":True,  "desc":"Fast coding model. Great for most programming tasks."},
    {"id":"unsloth/Qwen2.5-Coder-1.5B-Instruct",        "name":"Qwen2.5-Coder 1.5B",      "cat":"coding",    "params":"1.5B", "star":False, "desc":"Smallest coding model. Very fast training."},
    {"id":"unsloth/deepseek-coder-6.7b-instruct",        "name":"DeepSeek-Coder 6.7B",     "cat":"coding",    "params":"6.7B", "star":True,  "desc":"Excellent code generation and debugging."},
    {"id":"unsloth/deepseek-coder-1.3b-instruct",        "name":"DeepSeek-Coder 1.3B",     "cat":"coding",    "params":"1.3B", "star":False, "desc":"Tiny but capable coder."},
    {"id":"codellama/CodeLlama-7b-Instruct-hf",          "name":"CodeLlama 7B",            "cat":"coding",    "params":"7B",   "star":False, "desc":"Meta's dedicated code model. Strong Python/C++."},
    {"id":"codellama/CodeLlama-13b-Instruct-hf",         "name":"CodeLlama 13B",           "cat":"coding",    "params":"13B",  "star":False, "desc":"Larger CodeLlama. Better code understanding."},
    {"id":"bigcode/starcoder2-7b",                       "name":"StarCoder2 7B",           "cat":"coding",    "params":"7B",   "star":False, "desc":"Trained on 600+ languages from The Stack v2."},
    {"id":"bigcode/starcoder2-3b",                       "name":"StarCoder2 3B",           "cat":"coding",    "params":"3B",   "star":False, "desc":"Fast StarCoder model."},
    {"id":"Salesforce/codegen25-7B-instruct",            "name":"CodeGen2.5 7B",           "cat":"coding",    "params":"7B",   "star":False, "desc":"Salesforce code generation model."},
    # ── REASONING ─────────────────────────────────────────────────────────────
    {"id":"unsloth/DeepSeek-R1-Distill-Qwen-7B",         "name":"DeepSeek-R1 7B",          "cat":"reasoning", "params":"7B",   "star":True,  "desc":"Best reasoning. Built-in chain-of-thought. Think before answering."},
    {"id":"unsloth/DeepSeek-R1-Distill-Qwen-1.5B",       "name":"DeepSeek-R1 1.5B",        "cat":"reasoning", "params":"1.5B", "star":False, "desc":"Smallest R1 with reasoning. Fast training."},
    {"id":"unsloth/DeepSeek-R1-Distill-Llama-8B",        "name":"DeepSeek-R1 Llama 8B",    "cat":"reasoning", "params":"8B",   "star":False, "desc":"R1 reasoning on Llama 3 base. Strong quality."},
    {"id":"Qwen/QwQ-32B-Preview",                        "name":"QwQ 32B",                 "cat":"reasoning", "params":"32B",  "star":False, "desc":"Qwen's reasoning model. Very powerful."},
    # ── GENERAL — QWEN ────────────────────────────────────────────────────────
    {"id":"unsloth/Qwen2.5-7B-Instruct",                 "name":"Qwen2.5 7B",              "cat":"general",   "params":"7B",   "star":True,  "desc":"Best all-around small model. Excellent on everything."},
    {"id":"unsloth/Qwen2.5-3B-Instruct",                 "name":"Qwen2.5 3B",              "cat":"general",   "params":"3B",   "star":False, "desc":"Fast and capable. Good balance of speed/quality."},
    {"id":"unsloth/Qwen2.5-1.5B-Instruct",               "name":"Qwen2.5 1.5B",            "cat":"general",   "params":"1.5B", "star":False, "desc":"Tiny model. Fastest training."},
    {"id":"Qwen/Qwen2.5-14B-Instruct",                   "name":"Qwen2.5 14B",             "cat":"general",   "params":"14B",  "star":False, "desc":"Larger Qwen2.5. Better quality."},
    {"id":"Qwen/Qwen2.5-32B-Instruct",                   "name":"Qwen2.5 32B",             "cat":"general",   "params":"32B",  "star":False, "desc":"Very large Qwen2.5. Near GPT-4 quality."},
    {"id":"Qwen/Qwen2.5-72B-Instruct",                   "name":"Qwen2.5 72B",             "cat":"general",   "params":"72B",  "star":False, "desc":"Largest Qwen2.5. Best quality."},
    # ── GENERAL — LLAMA ───────────────────────────────────────────────────────
    {"id":"unsloth/Llama-3.2-3B-Instruct",               "name":"Llama 3.2 3B",            "cat":"general",   "params":"3B",   "star":True,  "desc":"Meta's latest small model. Very reliable."},
    {"id":"unsloth/Llama-3.2-1B-Instruct",               "name":"Llama 3.2 1B",            "cat":"general",   "params":"1B",   "star":False, "desc":"Smallest Llama. Ultra fast training."},
    {"id":"unsloth/Meta-Llama-3.1-8B-Instruct",          "name":"Llama 3.1 8B",            "cat":"general",   "params":"8B",   "star":True,  "desc":"Llama 3.1 flagship small model. Excellent quality."},
    {"id":"meta-llama/Llama-3.1-70B-Instruct",           "name":"Llama 3.1 70B",           "cat":"general",   "params":"70B",  "star":False, "desc":"Large Llama. Very high quality. Needs A100."},
    {"id":"meta-llama/Meta-Llama-3-8B-Instruct",         "name":"Llama 3 8B",              "cat":"general",   "params":"8B",   "star":False, "desc":"Previous Llama 3 generation."},
    {"id":"unsloth/tinyllama-1.1b-chat-v1.0",            "name":"TinyLlama 1.1B",          "cat":"general",   "params":"1.1B", "star":False, "desc":"Ultra small. Fastest possible training."},
    # ── GENERAL — MISTRAL ─────────────────────────────────────────────────────
    {"id":"unsloth/mistral-7b-instruct-v0.3",            "name":"Mistral 7B v0.3",         "cat":"general",   "params":"7B",   "star":True,  "desc":"Classic reliable model. Great for documents."},
    {"id":"unsloth/Mistral-Nemo-Instruct-2407",          "name":"Mistral Nemo 12B",        "cat":"general",   "params":"12B",  "star":False, "desc":"Mistral's 12B model. Strong quality."},
    {"id":"mistralai/Mixtral-8x7B-Instruct-v0.1",        "name":"Mixtral 8x7B MoE",        "cat":"general",   "params":"47B",  "star":False, "desc":"Mixture of experts. Very high quality."},
    {"id":"mistralai/Mistral-7B-v0.1",                   "name":"Mistral 7B v0.1 (base)",  "cat":"general",   "params":"7B",   "star":False, "desc":"Base Mistral. Fine-tune from scratch."},
    # ── GENERAL — PHI ─────────────────────────────────────────────────────────
    {"id":"unsloth/Phi-3-mini-4k-instruct",              "name":"Phi-3 Mini 3.8B",         "cat":"general",   "params":"3.8B", "star":True,  "desc":"Microsoft model. Best factual accuracy for its size."},
    {"id":"unsloth/Phi-3.5-mini-instruct",               "name":"Phi-3.5 Mini 3.8B",       "cat":"general",   "params":"3.8B", "star":False, "desc":"Phi-3.5 with improved capabilities."},
    {"id":"microsoft/phi-4",                             "name":"Phi-4 14B",               "cat":"general",   "params":"14B",  "star":False, "desc":"Microsoft Phi-4. Very strong quality."},
    {"id":"microsoft/Phi-3-medium-4k-instruct",          "name":"Phi-3 Medium 14B",        "cat":"general",   "params":"14B",  "star":False, "desc":"Larger Phi-3."},
    # ── GENERAL — GEMMA ───────────────────────────────────────────────────────
    {"id":"unsloth/gemma-2-2b-it",                       "name":"Gemma 2 2B",              "cat":"general",   "params":"2B",   "star":False, "desc":"Google Gemma 2. Very efficient small model."},
    {"id":"unsloth/gemma-2-9b-it",                       "name":"Gemma 2 9B",              "cat":"general",   "params":"9B",   "star":False, "desc":"Gemma 2 medium. Strong all-around."},
    {"id":"google/gemma-2-27b-it",                       "name":"Gemma 2 27B",             "cat":"general",   "params":"27B",  "star":False, "desc":"Large Gemma 2. High quality."},
    {"id":"google/gemma-7b-it",                          "name":"Gemma 1 7B",              "cat":"general",   "params":"7B",   "star":False, "desc":"Original Gemma 7B."},
    # ── GENERAL — FALCON ──────────────────────────────────────────────────────
    {"id":"tiiuae/falcon-7b-instruct",                   "name":"Falcon 7B",               "cat":"general",   "params":"7B",   "star":False, "desc":"TII Falcon. Strong multilingual."},
    {"id":"tiiuae/falcon-40b-instruct",                  "name":"Falcon 40B",              "cat":"general",   "params":"40B",  "star":False, "desc":"Large Falcon. High quality."},
    # ── GENERAL — INTERNLM ────────────────────────────────────────────────────
    {"id":"internlm/internlm2_5-7b-chat",                "name":"InternLM 2.5 7B",         "cat":"general",   "params":"7B",   "star":False, "desc":"Shanghai AI Lab. Strong on Chinese + English."},
    {"id":"internlm/internlm2_5-20b-chat",               "name":"InternLM 2.5 20B",        "cat":"general",   "params":"20B",  "star":False, "desc":"Larger InternLM. Very capable."},
    # ── GENERAL — YI ──────────────────────────────────────────────────────────
    {"id":"01-ai/Yi-1.5-6B-Chat",                        "name":"Yi 1.5 6B",               "cat":"general",   "params":"6B",   "star":False, "desc":"01.AI Yi model. Strong reasoning."},
    {"id":"01-ai/Yi-1.5-34B-Chat",                       "name":"Yi 1.5 34B",              "cat":"general",   "params":"34B",  "star":False, "desc":"Large Yi. Near GPT-4 quality."},
    # ── CHAT / CONVERSATION ───────────────────────────────────────────────────
    {"id":"HuggingFaceH4/zephyr-7b-beta",                "name":"Zephyr 7B Beta",          "cat":"chat",      "params":"7B",   "star":False, "desc":"Great for conversations. Fine-tuned Mistral."},
    {"id":"openchat/openchat-3.5-0106",                  "name":"OpenChat 3.5",            "cat":"chat",      "params":"7B",   "star":False, "desc":"Best open chat model. GPT-3.5 level."},
    {"id":"teknium/OpenHermes-2.5-Mistral-7B",           "name":"OpenHermes 2.5",          "cat":"chat",      "params":"7B",   "star":False, "desc":"Strong instruction following."},
    {"id":"NousResearch/Hermes-3-Llama-3.1-8B",          "name":"Hermes 3 Llama 8B",       "cat":"chat",      "params":"8B",   "star":False, "desc":"Nous Research. Excellent chat and reasoning."},
    {"id":"unsloth/zephyr-sft-bnb-4bit",                 "name":"Zephyr SFT 4bit",         "cat":"chat",      "params":"7B",   "star":False, "desc":"Optimized Zephyr for fast training."},
    # ── MULTILINGUAL ──────────────────────────────────────────────────────────
    {"id":"unsloth/Qwen2.5-3B-Instruct",                 "name":"Qwen2.5 3B (multilingual)","cat":"multilingual","params":"3B", "star":True,  "desc":"Best free multilingual model. 29 languages."},
    {"id":"CohereForAI/aya-23-8B",                       "name":"Aya 23 8B",               "cat":"multilingual","params":"8B", "star":False, "desc":"65 languages. Purpose-built multilingual."},
    {"id":"CohereForAI/aya-23-35B",                      "name":"Aya 23 35B",              "cat":"multilingual","params":"35B","star":False, "desc":"Large multilingual Aya model."},
    {"id":"saillab/Sailor-7B-Chat",                      "name":"Sailor 7B",               "cat":"multilingual","params":"7B", "star":False, "desc":"Southeast Asian languages specialist."},
    # ── MEDICAL ───────────────────────────────────────────────────────────────
    {"id":"unsloth/Phi-3-mini-4k-instruct",              "name":"Phi-3 Mini (medical)",    "cat":"medical",   "params":"3.8B", "star":True,  "desc":"Best factual accuracy for medical content."},
    {"id":"BioMistral/BioMistral-7B",                    "name":"BioMistral 7B",           "cat":"medical",   "params":"7B",   "star":False, "desc":"Fine-tuned on biomedical literature."},
    {"id":"medalpaca/medalpaca-7b",                      "name":"MedAlpaca 7B",            "cat":"medical",   "params":"7B",   "star":False, "desc":"Medical fine-tuned LLaMA."},
    # ── LEGAL ─────────────────────────────────────────────────────────────────
    {"id":"unsloth/mistral-7b-instruct-v0.3",            "name":"Mistral 7B (legal)",      "cat":"legal",     "params":"7B",   "star":True,  "desc":"Best for long legal documents."},
    {"id":"AdaptLLM/law-LLM",                            "name":"Law-LLM",                 "cat":"legal",     "params":"7B",   "star":False, "desc":"Pre-trained on legal corpus."},
]


class ModelResolver:
    """
    Resolve any HuggingFace model ID into a complete training config.
    Works with Unsloth models (faster) and any standard model.
    """

    def resolve(self, model_id: str, hf_token: str = None) -> Dict:
        """
        Given any model ID, return complete training config.
        Tries HF API first, falls back to pattern matching.
        """
        model_id = model_id.strip()

        # Check popular models first (instant)
        for m in POPULAR_MODELS:
            if m["id"].lower() == model_id.lower():
                arch = self._detect_arch(model_id, [])
                return self._build_config(model_id, m.get("params","7B"), True, arch=arch)

        # Try HF API
        api_info = self._fetch_hf_info(model_id, hf_token)
        if api_info:
            return self._config_from_api(model_id, api_info)

        # Fall back to pattern matching
        return self._config_from_pattern(model_id)

    def search(self, query: str, hf_token: str = None, limit: int = 20) -> List[Dict]:
        """
        Search models. Always searches local registry first (instant),
        then enriches with HF API results if available.
        """
        # Always search local registry first — instant, no internet needed
        local = self._search_popular(query)

        # Try HF API to get more results
        api_results = []
        try:
            params = {
                "search":    query,
                "limit":     limit,
                "sort":      "downloads",
                "direction": "-1",
                "filter":    "text-generation",
            }
            headers = dict(HEADERS)
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            r = requests.get(f"{HF_API}/models", params=params,
                             headers=headers, timeout=6)

            if r.status_code == 200:
                for m in r.json()[:limit]:
                    mid     = m.get("id","")
                    tags    = m.get("tags",[]) or []
                    private = m.get("private", False)
                    gated   = m.get("gated", False)
                    if private: continue
                    # Skip if already in local results
                    if any(l["id"] == mid for l in local):
                        continue
                    api_results.append({
                        "id":        mid,
                        "name":      mid.split("/")[-1],
                        "params":    self._estimate_params(mid, tags),
                        "arch":      self._detect_arch(mid, tags),
                        "unsloth":   self._is_unsloth(mid),
                        "downloads": m.get("downloads", 0),
                        "likes":     m.get("likes", 0),
                        "gated":     gated,
                        "tags":      tags[:5],
                        "author":    mid.split("/")[0] if "/" in mid else "",
                        "desc":      "",
                    })
        except Exception as e:
            logger.debug(f"HF API unavailable: {e} — using local registry only")

        # Combine: local first (curated), then API results
        combined = local + api_results
        return combined[:limit]

    def _search_popular(self, query: str) -> List[Dict]:
        """Search built-in registry — works 100% offline."""
        q = query.lower().strip()
        if not q:
            return []
        results = []
        for m in POPULAR_MODELS:
            score = 0
            mid   = m["id"].lower()
            name  = m["name"].lower()
            desc  = m.get("desc","").lower()
            cat   = m.get("cat","")

            # Score matches
            if q in mid:         score += 3
            if q in name:        score += 3
            if q in cat:         score += 2
            if q in desc:        score += 1
            if m.get("star"):    score += 1

            # Also match partial words
            for word in q.split():
                if len(word) > 2:
                    if word in mid:   score += 2
                    if word in name:  score += 2
                    if word in desc:  score += 1

            if score > 0:
                results.append({
                    "id":      m["id"],
                    "name":    m["name"],
                    "params":  m["params"],
                    "arch":    self._detect_arch(m["id"], []),
                    "unsloth": self._is_unsloth(m["id"]),
                    "star":    m.get("star", False),
                    "cat":     cat,
                    "desc":    m.get("desc",""),
                    "score":   score,
                    "downloads": 999999 if m.get("star") else 100000,
                })

        # Sort by score (best match first)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:20]

    def get_popular(self, category: str = "all") -> List[Dict]:
        """Return popular models, optionally filtered by category."""
        if category == "all":
            return POPULAR_MODELS
        return [m for m in POPULAR_MODELS if m.get("cat") == category]

    def _fetch_hf_info(self, model_id: str, token: str = None) -> Optional[Dict]:
        """Fetch model card info from HF API."""
        try:
            headers = dict(HEADERS)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            r = requests.get(f"{HF_API}/models/{model_id}",
                             headers=headers, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _config_from_api(self, model_id: str, api_info: Dict) -> Dict:
        """Build config from HF API response."""
        tags      = api_info.get("tags", []) or []
        siblings  = api_info.get("siblings", []) or []
        safetensors = any("safetensors" in s.get("rfilename","") for s in siblings)
        config    = api_info.get("config", {}) or {}
        arch_list = config.get("architectures", [])
        arch      = self._detect_arch(model_id, tags + [str(a) for a in arch_list])
        params    = self._estimate_params(model_id, tags)
        gated     = api_info.get("gated", False)
        return self._build_config(model_id, params, self._is_unsloth(model_id),
                                  arch=arch, gated=gated,
                                  safetensors=safetensors)

    def _config_from_pattern(self, model_id: str) -> Dict:
        """Build config from model ID pattern matching."""
        arch   = self._detect_arch(model_id, [])
        params = self._estimate_params(model_id, [])
        return self._build_config(model_id, params, self._is_unsloth(model_id), arch=arch)

    def _build_config(self, model_id: str, params: str, unsloth: bool,
                      arch: str = "default", gated: bool = False,
                      safetensors: bool = True) -> Dict:
        """Build complete training config for a model."""
        arch_cfg  = ARCH_CONFIGS.get(arch, ARCH_CONFIGS["default"])
        param_num = self._params_to_num(params)
        size      = "small" if param_num < 3 else ("medium" if param_num < 6 else "large")

        lora_r = {"small": 16, "medium": 16, "large": 8}[size]
        lr     = {"small": 3e-4, "medium": 2e-4, "large": 1e-4}[size]
        batch  = {"small": 4, "medium": 2, "large": 1}[size]
        accum  = {"small": 2, "medium": 4, "large": 8}[size]
        seq    = {"small": 2048, "medium": 2048, "large": 1024}[size]
        epochs = {"small": 4, "medium": 3, "large": 3}[size]
        colab_min = {"small": 20, "medium": 45, "large": 85}[size]
        gguf_gb   = {"small": 1.2, "medium": 2.5, "large": 5.0}[size]

        return {
            "id":             model_id,
            "name":           model_id.split("/")[-1],
            "author":         model_id.split("/")[0] if "/" in model_id else "unknown",
            "params":         params,
            "size":           size,
            "arch":           arch,
            "unsloth":        unsloth,
            "gated":          gated,
            "safetensors":    safetensors,
            "target_modules": arch_cfg["target_modules"],
            "chat_template":  arch_cfg["template"],
            "chat_format":    arch_cfg["chat_format"],
            "eos_token":      arch_cfg["eos"],
            "lora_r":         lora_r,
            "lora_alpha":     lora_r * 2 if size != "large" else lora_r,
            "learning_rate":  lr,
            "batch_size":     batch,
            "grad_accum":     accum,
            "max_seq":        seq,
            "epochs":         epochs,
            "colab_min":      colab_min,
            "gguf_gb":        gguf_gb,
            "hf_url":         f"https://huggingface.co/{model_id}",
        }

    def _detect_arch(self, model_id: str, tags: List[str]) -> str:
        """Detect model architecture from ID and tags."""
        combined = (model_id + " " + " ".join(tags)).lower()
        # Check most specific patterns first
        if any(x in combined for x in ["deepseek"]):
            return "deepseek"
        if any(x in combined for x in ["qwen"]):
            return "qwen2"
        if any(x in combined for x in ["gemma"]):
            return "gemma"
        if any(x in combined for x in ["phi-3","phi3","phi-4","phi4"]):
            return "phi"
        if any(x in combined for x in ["mistral","mixtral","nemo"]):
            return "mistral"
        if any(x in combined for x in ["llama","alpaca","vicuna","wizard","codellama","tinyllama"]):
            return "llama"
        if any(x in combined for x in ["falcon"]):
            return "falcon"
        if any(x in combined for x in ["gpt-neox","pythia","dolly"]):
            return "gpt_neox"
        if any(x in combined for x in ["t5","flan","mt5"]):
            return "t5"
        if any(x in combined for x in ["starcoder","santacoder"]):
            return "qwen2"  # StarCoder2 uses similar arch
        if any(x in combined for x in ["internlm"]):
            return "llama"
        if any(x in combined for x in ["yi-"]):
            return "llama"
        return "default"

    def _is_unsloth(self, model_id: str) -> bool:
        mid = model_id.lower()
        if mid.startswith("unsloth/"):
            return True
        return any(x in mid for x in UNSLOTH_SUPPORTED)

    def _estimate_params(self, model_id: str, tags: List[str]) -> str:
        """Estimate parameter count from model ID."""
        combined = (model_id + " " + " ".join(tags)).lower()
        patterns = [
            (r'(\d+\.?\d*)b', lambda m: f"{m.group(1)}B"),
            (r'(\d+)m\b',     lambda m: f"{int(m.group(1))/1000:.1f}B"),
            (r'-(\d+)-',      lambda m: f"{m.group(1)}B"),
        ]
        for pattern, fmt in patterns:
            m = re.search(pattern, combined)
            if m:
                try:
                    return fmt(m)
                except Exception:
                    continue
        # Default guesses based on known names
        if any(x in combined for x in ["tiny","mini","small","1b","1.1b","1.5b"]):
            return "1.5B"
        if any(x in combined for x in ["3b","3.8b","4b"]):
            return "3B"
        if any(x in combined for x in ["7b","8b","6.7b"]):
            return "7B"
        if any(x in combined for x in ["13b","14b"]):
            return "13B"
        if any(x in combined for x in ["70b","72b"]):
            return "70B"
        return "7B"  # default assumption

    def _params_to_num(self, params: str) -> float:
        try:
            return float(re.sub(r'[Bb]','', params))
        except Exception:
            return 7.0
