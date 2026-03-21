# core/hf_streamer.py — Stream data from HuggingFace datasets
# No downloading. Streams only what you need. Free.

import logging
from typing import List, Dict, Optional
import itertools

logger = logging.getLogger(__name__)

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# All free, public, streamable datasets organized by category

HF_DATASETS = {

    "coding": {
        "name": "Coding / Programming",
        "icon": "💻",
        "datasets": [
            {
                "id":          "codeparrot/github-code",
                "name":        "GitHub Code (115 languages)",
                "description": "Real code from GitHub. Best for coding assistants.",
                "size":        "~1TB total, stream any amount",
                "fields":      {"text": "code", "lang_filter": "programming_language"},
                "languages":   ["Python","JavaScript","TypeScript","Java","C","C++",
                               "C#","Go","Rust","PHP","Ruby","Swift","Kotlin","SQL",
                               "Shell","HTML","CSS","R","Scala","Dart"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "bigcode/bigcodebench",
                "name":        "BigCodeBench",
                "description": "Hard coding tasks with unit tests. High quality.",
                "size":        "~1.1K problems",
                "fields":      {"text": "complete_prompt"},
                "languages":   ["Python"],
                "config":      None,
                "split":       "v0.1.2",
                "recommended": False,
            },
            {
                "id":          "iamtarun/python_code_instructions_18k_alpaca",
                "name":        "Python Instructions 18K",
                "description": "18K Python coding instructions in Alpaca format.",
                "size":        "18K samples",
                "fields":      {"instruction": "instruction", "output": "output"},
                "languages":   ["Python"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
            {
                "id":          "sahil2801/CodeAlpaca-20k",
                "name":        "CodeAlpaca 20K",
                "description": "20K coding instruction-following examples.",
                "size":        "20K samples",
                "fields":      {"instruction": "instruction", "output": "output"},
                "languages":   ["Python","JavaScript","General"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "chat": {
        "name": "Chat / Conversation",
        "icon": "💬",
        "datasets": [
            {
                "id":          "HuggingFaceH4/ultrachat_200k",
                "name":        "UltraChat 200K",
                "description": "200K high-quality multi-turn conversations. Best for chatbots.",
                "size":        "200K conversations",
                "fields":      {"messages": "messages"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train_sft",
                "recommended": True,
            },
            {
                "id":          "tatsu-lab/alpaca",
                "name":        "Stanford Alpaca 52K",
                "description": "52K instruction-following pairs. Classic fine-tuning dataset.",
                "size":        "52K samples",
                "fields":      {"instruction": "instruction", "output": "output"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
            {
                "id":          "Open-Orca/OpenOrca",
                "name":        "OpenOrca",
                "description": "High quality reasoning + chat pairs. GPT-4 level.",
                "size":        "~1M samples, stream any amount",
                "fields":      {"instruction": "question", "output": "response"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "reasoning": {
        "name": "Reasoning / Math",
        "icon": "🧠",
        "datasets": [
            {
                "id":          "microsoft/orca-math-word-problems-200k",
                "name":        "Orca Math 200K",
                "description": "200K math word problems with reasoning steps.",
                "size":        "200K samples",
                "fields":      {"instruction": "question", "output": "answer"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "lighteval/MATH",
                "name":        "MATH Dataset",
                "description": "Competition math problems with solutions.",
                "size":        "12.5K problems",
                "fields":      {"instruction": "problem", "output": "solution"},
                "languages":   ["English"],
                "config":      "all",
                "split":       "train",
                "recommended": False,
            },
            {
                "id":          "reasoning-machines/gsm8k",
                "name":        "GSM8K",
                "description": "Grade school math with step-by-step reasoning.",
                "size":        "8.5K problems",
                "fields":      {"instruction": "question", "output": "answer"},
                "languages":   ["English"],
                "config":      "main",
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "medical": {
        "name": "Medical / Healthcare",
        "icon": "🏥",
        "datasets": [
            {
                "id":          "medalpaca/medical_meadow_medqa",
                "name":        "Medical MedQA",
                "description": "Medical Q&A from USMLE exam questions.",
                "size":        "10K questions",
                "fields":      {"instruction": "input", "output": "output"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "medalpaca/medical_meadow_wikidoc",
                "name":        "Medical WikiDoc",
                "description": "Medical knowledge from WikiDoc. Great for clinical AI.",
                "size":        "67K entries",
                "fields":      {"instruction": "input", "output": "output"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "legal": {
        "name": "Legal / Law",
        "icon": "⚖️",
        "datasets": [
            {
                "id":          "pile-of-law/pile-of-law",
                "name":        "Pile of Law",
                "description": "Large collection of legal texts. Contracts, cases, regulations.",
                "size":        "~256GB total, stream any amount",
                "fields":      {"text": "text"},
                "languages":   ["English"],
                "config":      "courtlistener_opinions",
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "nguyen-brat/legal-data",
                "name":        "Legal QA Data",
                "description": "Legal Q&A pairs for fine-tuning.",
                "size":        "Small",
                "fields":      {"instruction": "question", "output": "answer"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "science": {
        "name": "Science / Research",
        "icon": "🔬",
        "datasets": [
            {
                "id":          "allenai/peS2o",
                "name":        "peS2o (Scientific Papers)",
                "description": "Cleaned scientific papers. Great for research AI.",
                "size":        "~40GB, stream any amount",
                "fields":      {"text": "text"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "ccdv/arxiv-summarization",
                "name":        "ArXiv Summarization",
                "description": "ArXiv papers with abstracts. Good for science Q&A.",
                "size":        "215K papers",
                "fields":      {"instruction": "article", "output": "abstract"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "general": {
        "name": "General Knowledge",
        "icon": "✨",
        "datasets": [
            {
                "id":          "wikimedia/wikipedia",
                "name":        "Wikipedia",
                "description": "Full Wikipedia. Any language. Best general knowledge.",
                "size":        "~20GB per language, stream any amount",
                "fields":      {"text": "text"},
                "languages":   ["en","hi","fr","de","es","ja","zh","ar","pt","ru"],
                "config":      "20231101.en",
                "split":       "train",
                "recommended": True,
            },
            {
                "id":          "c4",
                "name":        "C4 (Colossal Clean Crawled Corpus)",
                "description": "Massive cleaned web text. Best for general language.",
                "size":        "~750GB, stream any amount",
                "fields":      {"text": "text"},
                "languages":   ["English"],
                "config":      "en",
                "split":       "train",
                "recommended": False,
            },
        ],
    },

    "finance": {
        "name": "Finance / Business",
        "icon": "💹",
        "datasets": [
            {
                "id":          "FinGPT/fingpt-sentiment-train",
                "name":        "FinGPT Sentiment",
                "description": "Financial news sentiment analysis pairs.",
                "size":        "76K samples",
                "fields":      {"instruction": "input", "output": "output"},
                "languages":   ["English"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
        ],
    },

    "multilingual": {
        "name": "Multilingual",
        "icon": "🌍",
        "datasets": [
            {
                "id":          "CohereForAI/aya_dataset",
                "name":        "Aya Dataset",
                "description": "65 languages, 204K instruction pairs. Best multilingual.",
                "size":        "204K samples",
                "fields":      {"instruction": "inputs", "output": "targets"},
                "languages":   ["65 languages"],
                "config":      None,
                "split":       "train",
                "recommended": True,
            },
        ],
    },
}


class HFStreamer:
    """
    Stream samples from any HuggingFace public dataset.
    No full download. Fetches only what you need.
    """

    def stream(self, dataset_id: str, config: Optional[str],
               split: str, fields: Dict, n_samples: int,
               lang_filter: Optional[str] = None,
               hf_token: Optional[str] = None,
               progress_callback=None) -> List[Dict]:
        """
        Stream n_samples from a HuggingFace dataset.
        Supports gated/private datasets via HF token.
        Returns list of chunks ready for training.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")

        logger.info(f"Streaming {n_samples} samples from {dataset_id}")

        # Login with token if provided
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
                logger.info("HuggingFace token applied")
            except Exception as e:
                logger.warning(f"HF login warning: {e}")

        # Some datasets use data_dir instead of name/config
        # e.g. bigcode/starcoderdata uses data_dir="python" not name="python"
        load_kwargs = {
            "path":      dataset_id,
            "split":     split,
            "streaming": True,
        }
        if hf_token:
            load_kwargs["token"] = hf_token

        # Add config — try as name first, fall back to data_dir
        if config and config not in ("default", "None", ""):
            load_kwargs["name"] = config

        try:
            ds = load_dataset(**load_kwargs)
        except (TypeError, ValueError) as e:
            err_s = str(e)
            if "BuilderConfig" in err_s or "not found" in err_s or "name" in err_s:
                # Config name not found — try as data_dir instead
                load_kwargs.pop("name", None)
                if config and config not in ("default", "None", ""):
                    load_kwargs["data_dir"] = config
                try:
                    ds = load_dataset(**load_kwargs)
                except Exception:
                    # Last resort — no config at all
                    load_kwargs.pop("data_dir", None)
                    ds = load_dataset(**load_kwargs)
            else:
                raise
        except RuntimeError as e:
            err_str = str(e)
            # Dataset uses legacy loading script — no longer supported
            if "Dataset scripts are no longer supported" in err_str or "loading script" in err_str:
                raise ValueError(
                    f"❌ Dataset '{dataset_id}' uses a legacy loading script "
                    f"which is no longer supported by HuggingFace datasets >= 4.5.0. "
                    f"Please use an alternative dataset. "
                    f"For coding: try bigcode/starcoderdata or nampdn-ai/tiny-codes instead."
                )
            raise
        except Exception as e:
            err = str(e).lower()
            if "gated" in err or "access" in err or "401" in err:
                raise PermissionError(
                    f"Dataset '{dataset_id}' requires a HuggingFace token. "
                    f"Add your token in the HF Datasets tab."
                )
            if "dataset scripts are no longer supported" in err or "loading script" in err:
                raise ValueError(
                    f"❌ '{dataset_id}' uses a legacy script. "
                    f"Try: bigcode/starcoderdata or nampdn-ai/tiny-codes"
                )
            logger.error(f"Failed to load {dataset_id}: {e}")
            raise

        chunks  = []
        fetched = 0

        for i, sample in enumerate(itertools.islice(ds, n_samples * 3)):
            # Language filter for code datasets
            if lang_filter and lang_filter in sample:
                if sample[lang_filter] != lang_filter:
                    continue

            text = self._extract_text(sample, fields)
            if not text or len(text.strip()) < 50:
                continue

            chunks.append({
                "text":        text[:3000],  # cap at 3000 chars
                "source":      dataset_id.split("/")[-1],
                "source_type": self._detect_type(fields),
            })
            fetched += 1

            if progress_callback and fetched % 100 == 0:
                progress_callback(fetched, n_samples, dataset_id)

            if fetched >= n_samples:
                break

        logger.info(f"Streamed {len(chunks)} chunks from {dataset_id}")
        return chunks

    def _extract_text(self, sample: Dict, fields: Dict) -> str:
        """Extract text from sample based on field mapping."""
        # Direct text field
        if "text" in fields and fields["text"] in sample:
            return str(sample[fields["text"]] or "")

        # Instruction + output pair
        if "instruction" in fields and "output" in fields:
            inst = str(sample.get(fields["instruction"], "") or "")
            out  = str(sample.get(fields["output"], "") or "")
            if inst and out:
                return f"Question: {inst}\n\nAnswer: {out}"
            return inst or out

        # Messages format (chat)
        if "messages" in fields and fields["messages"] in sample:
            msgs = sample[fields["messages"]] or []
            parts = []
            for msg in msgs:
                role    = msg.get("role","")
                content = msg.get("content","") or ""
                if role and content:
                    parts.append(f"{role.upper()}: {content}")
            return "\n\n".join(parts)

        # Fallback — try common field names
        for key in ["text","content","code","body","passage","document","article"]:
            if key in sample and sample[key]:
                return str(sample[key])

        return ""

    def _detect_type(self, fields: Dict) -> str:
        if "lang_filter" in fields or "code" in str(fields):
            return "code"
        if "instruction" in fields:
            return "notes"
        return "document"


    def stream_via_api(self, dataset_id: str, config: str,
                       split: str, n_samples: int,
                       hf_token: str = None,
                       progress_callback=None) -> List[Dict]:
        """
        Stream rows directly using HuggingFace Datasets Server API.
        Works with gated datasets when token is provided.
        Uses: datasets-server.huggingface.co/rows
        """
        import requests

        base_url = "https://datasets-server.huggingface.co/rows"
        headers  = {"User-Agent": "PersonalForge/9.0"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        chunks       = []
        offset       = 0
        batch_size   = 100  # max per API call
        fetched      = 0

        # First — get available splits
        try:
            splits_url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_id}"
            sr = requests.get(splits_url, headers=headers, timeout=10)
            if sr.status_code == 200:
                available = [s["split"] for s in sr.json().get("splits", [])]
                if split not in available and available:
                    split = available[0]
                    logger.info(f"Using split: {split}")
        except Exception:
            pass

        logger.info(f"API streaming {n_samples} rows from {dataset_id}/{config}/{split}")

        while fetched < n_samples:
            batch = min(batch_size, n_samples - fetched)
            params = {
                "dataset": dataset_id,
                "config":  config or "default",
                "split":   split,
                "offset":  offset,
                "length":  batch,
            }
            try:
                r = requests.get(base_url, headers=headers, params=params, timeout=20)

                if r.status_code == 401:
                    raise PermissionError("Token required or invalid — add your HF token")
                if r.status_code == 403:
                    raise PermissionError("Access denied — accept dataset terms on HuggingFace first")
                if r.status_code == 404:
                    raise ValueError(f"Dataset not found: {dataset_id}")
                if r.status_code != 200:
                    raise Exception(f"API error {r.status_code}: {r.text[:200]}")

                data = r.json()
                rows = data.get("rows", [])
                if not rows:
                    break  # no more data

                for row_obj in rows:
                    row  = row_obj.get("row", {})
                    text = self._extract_text_from_row(row)
                    if text and len(text.strip()) > 50:
                        chunks.append({
                            "text":        text[:3000],
                            "source":      dataset_id.split("/")[-1],
                            "source_type": self._detect_type_from_row(row),
                        })
                        fetched += 1

                offset += batch
                if progress_callback:
                    progress_callback(fetched, n_samples, dataset_id)

                # Stop if we got everything
                if len(rows) < batch:
                    break

            except (PermissionError, ValueError) as e:
                raise
            except Exception as e:
                err = str(e)
                if "script" in err.lower() or "no longer supported" in err.lower():
                    raise ValueError(f"Dataset uses legacy script — not supported. Use starcoderdata instead.")
                logger.error(f"API stream error at offset {offset}: {e}")
                break

        logger.info(f"API streamed {len(chunks)} chunks from {dataset_id}")
        return chunks

    def _extract_text_from_row(self, row: Dict) -> str:
        """Extract text from a dataset row — tries all common field names."""
        # Code datasets
        for key in ["content","code","text","body"]:
            if key in row and row[key] and isinstance(row[key], str):
                return row[key]
        # Instruction datasets
        inst = row.get("instruction","") or row.get("question","") or row.get("input","")
        out  = row.get("output","")  or row.get("answer","")  or row.get("response","")
        if inst and out:
            return f"Question: {inst}\n\nAnswer: {out}"
        if inst: return inst
        if out:  return out
        # Chat/messages
        msgs = row.get("messages","") or row.get("conversations","")
        if msgs and isinstance(msgs, list):
            return "\n\n".join(
                f"{m.get('role','').upper()}: {m.get('content','')}"
                for m in msgs if m.get("content")
            )
        # Fallback — first long string value
        for v in row.values():
            if isinstance(v, str) and len(v) > 50:
                return v
        return ""

    def _detect_type_from_row(self, row: Dict) -> str:
        if any(k in row for k in ["code","content","language","programming_language"]):
            return "code"
        if any(k in row for k in ["instruction","question","input"]):
            return "notes"
        return "document"

    def get_dataset_info(self, dataset_id: str, hf_token: str = None) -> Dict:
        """
        Get splits and configs for any dataset via API.
        Used to populate split/config dropdowns dynamically.
        """
        import requests
        headers = {"User-Agent": "PersonalForge/9.0"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        info = {"splits": [], "configs": [], "parquet": []}

        # Get splits
        try:
            r = requests.get(
                f"https://datasets-server.huggingface.co/splits?dataset={dataset_id}",
                headers=headers, timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                info["splits"]  = list(set(s["split"]  for s in data.get("splits",[])))
                info["configs"] = list(set(s["config"] for s in data.get("splits",[])))
        except Exception as e:
            info["splits_error"] = str(e)

        # Get parquet files
        try:
            r = requests.get(
                f"https://huggingface.co/api/datasets/{dataset_id}/parquet/default/train",
                headers=headers, timeout=8
            )
            if r.status_code == 200:
                info["parquet"] = r.json()
        except Exception:
            pass

        return info



    def validate_dataset(self, dataset_id: str, config: str = None,
                         token: str = None) -> dict:
        """
        Quick check if a dataset is loadable without streaming all data.
        Returns {"ok": bool, "error": str, "fields": list}
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return {"ok": False, "error": "datasets library not installed"}

        load_kwargs = {
            "path":      dataset_id,
            "split":     "train",
            "streaming": True,
        }
        if config:
            load_kwargs["name"] = config
        if token:
            load_kwargs["token"] = token

        try:
            ds = load_dataset(**load_kwargs)
            # Try to get one sample to verify
            sample = next(iter(ds))
            fields = list(sample.keys())
            return {"ok": True, "fields": fields, "sample_keys": fields[:5]}
        except RuntimeError as e:
            if "script" in str(e).lower():
                return {"ok": False,
                        "error": f"Legacy dataset script — not supported in datasets>=4.5. Use starcoderdata instead."}
            return {"ok": False, "error": str(e)[:200]}
        except PermissionError as e:
            return {"ok": False, "error": "Needs HF token — add it in the token box"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def get_categories(self) -> Dict:
        return HF_DATASETS

    def get_datasets_for_category(self, category: str) -> List[Dict]:
        return HF_DATASETS.get(category, {}).get("datasets", [])

    def estimate_time(self, n_samples: int) -> str:
        mins = max(1, n_samples // 1000)
        return f"~{mins} minute{'s' if mins > 1 else ''}"

    def validate_token(self, token: str) -> Dict:
        """
        Simple token save — just check format and save.
        Real validation happens at stream time when we actually use it.
        No network calls — no false rejections.
        """
        token = token.strip()
        if not token:
            return {"valid": False, "error": "Token is empty"}
        if not token.startswith("hf_"):
            return {"valid": False, "error": "HuggingFace tokens start with hf_"}
        if len(token) < 10:
            return {"valid": False, "error": "Token looks too short"}
        # Accept it — will fail at stream time if genuinely wrong
        return {"valid": True, "username": "HF User", "saved": True}

    def check_dataset_access(self, dataset_id: str, token: str = None) -> Dict:
        """Check if a dataset is accessible (public or with token)."""
        try:
            import requests
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            r = requests.get(
                f"https://huggingface.co/api/datasets/{dataset_id}",
                headers=headers, timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                return {
                    "accessible": True,
                    "gated":      data.get("gated", False),
                    "private":    data.get("private", False),
                    "downloads":  data.get("downloads", 0),
                    "description":data.get("description","")[:200],
                }
            elif r.status_code == 401:
                return {"accessible": False, "error": "Requires authentication — add your HF token"}
            elif r.status_code == 403:
                return {"accessible": False, "error": "Access denied — you need to accept terms on HuggingFace first"}
            elif r.status_code == 404:
                return {"accessible": False, "error": "Dataset not found — check the owner/dataset-name format"}
            return {"accessible": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"accessible": False, "error": str(e)}
