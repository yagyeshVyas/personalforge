# core/pair_generator.py — LLM-powered training pair generator v6
# Uses local Ollama to generate high-quality, diverse, natural Q&A pairs
# Falls back to template generation if Ollama not available
# Quality: LLM-generated >> template-generated

import re, json, random, os, logging, requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

# ─── 3 TRAINING MODES ─────────────────────────────────────────────────────────

MODES = {
    "developer": {
        "name":        "Developer / Coder",
        "icon":        "💻",
        "description": "Code examples, technical explanations, step-by-step solutions.",
        "system": """You are an expert software developer trained on specific programming resources.
Think step by step inside <think>...</think> before every answer.
Always: write working code with comments, explain step by step, mention best practices.
Only use knowledge from your training data. Never hallucinate APIs or functions.""",
        "llm_prompt": """You are creating training data for a coding AI.
Given this text from a programming resource, generate {n} diverse, natural questions
that a developer would actually ask. Make them specific to the content.

Text: {text}

Generate exactly {n} questions. Each on its own line. No numbering.
Mix question types: how-to, explain, code example, best practice, debug, compare.""",
    },
    "thinker": {
        "name":        "Deep Thinker",
        "icon":        "🧠",
        "description": "Multi-angle analysis, connects ideas, thorough answers.",
        "system": """You are a deep analytical thinker trained on specific knowledge.
Think deeply inside <think>...</think> before every answer.
Always: analyze from multiple angles, consider pros/cons, connect related concepts.
Only use knowledge from your training data.""",
        "llm_prompt": """You are creating training data for an analytical AI.
Given this text, generate {n} deep, thought-provoking questions
that require analysis and synthesis to answer well.

Text: {text}

Generate exactly {n} questions. Each on its own line. No numbering.
Mix: analytical, comparative, implication, critical evaluation questions.""",
    },
    "factual": {
        "name":        "Honest / Factual",
        "icon":        "✅",
        "description": "Always truthful, cites sources, admits gaps.",
        "system": """You are a precise factual assistant trained on specific documents.
Verify inside <think>...</think> before every answer.
Always: state facts clearly, cite which document, admit gaps honestly.
Never guess. Only use knowledge from your training data.""",
        "llm_prompt": """You are creating training data for a factual AI assistant.
Given this text, generate {n} clear, direct questions
that have specific factual answers in the text.

Text: {text}

Generate exactly {n} questions. Each on its own line. No numbering.
Focus on: definitions, specific facts, explanations, summaries.""",
    },
}

# Fallback templates if Ollama not available
FALLBACK_QUESTIONS = [
    "What is {t}?",
    "Explain {t} in detail.",
    "How does {t} work?",
    "What are the key points about {t}?",
    "Why is {t} important?",
    "Summarize {t}.",
    "Give me an example of {t}.",
    "What should I know about {t}?",
]

OUT_OF_SCOPE = [
    "What is the weather today?", "Who won the latest election?",
    "What is the current stock price?", "Tell me today's news.",
    "Who is the current president?", "What time is it?",
    "Tell me a joke.", "What is Bitcoin worth?",
    "How do I cook pasta?", "What is the capital of France?",
    "Who invented the telephone?", "What movies are popular?",
    "What happened in the news?", "Who is the richest person?",
    "What is the weather forecast?", "Recommend a restaurant.",
    "What is today's date?", "Tell me about recent events.",
    "Who won the World Cup?", "What is the latest iPhone?",
]

REFUSALS = [
    "I don't have that information in my training data.",
    "That topic isn't in my training data. Please ask about what I was trained on.",
    "I'm not able to answer that — it's outside my training data.",
    "My training data doesn't include that.",
]


class PairGenerator:

    def __init__(self, chunk_size=800, chunk_overlap=120):
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.ollama_model   = None
        self.ollama_available = False
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is running locally."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Prefer smaller fast models for pair generation
                preferred = ["llama3.2:1b","llama3.2:3b","qwen2.5:1.5b",
                             "phi3:mini","mistral:7b","llama3:8b"]
                for m in preferred:
                    if any(m in model for model in models):
                        self.ollama_model     = m
                        self.ollama_available = True
                        logger.info(f"Ollama available: {m}")
                        return
                if models:
                    self.ollama_model     = models[0]
                    self.ollama_available = True
                    logger.info(f"Ollama available: {models[0]}")
        except Exception:
            logger.info("Ollama not available — using template generation")

    def _generate_questions_with_llm(self, text: str, mode: str, n: int) -> List[str]:
        """Use local Ollama to generate high-quality questions."""
        if not self.ollama_available:
            return []

        prompt = MODES[mode]["llm_prompt"].format(
            text=text[:600],  # keep prompt manageable
            n=n
        )

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.8, "num_predict": 300}
                },
                timeout=30
            )
            if r.status_code == 200:
                response = r.json().get("response","")
                # Parse questions — one per line
                questions = [
                    line.strip()
                    for line in response.split('\n')
                    if line.strip() and len(line.strip()) > 15
                    and not line.strip().startswith('#')
                ]
                return questions[:n]
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
        return []

    def _generate_answer_with_llm(self, question: str, context: str, mode: str) -> Optional[str]:
        """Use Ollama to generate a high-quality answer."""
        if not self.ollama_available:
            return None

        prompt = f"""Based ONLY on the following text, answer this question accurately.
If the text doesn't contain the answer, say "I don't have that information."

Text: {context[:800]}

Question: {question}

Answer concisely and accurately:"""

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 400}
                },
                timeout=30
            )
            if r.status_code == 200:
                return r.json().get("response","").strip()
        except Exception:
            pass
        return None

    def generate(self, file_chunks: List[Dict], mode: str = "factual",
                 pairs_per_chunk: int = 4, max_pairs: int = 5000,
                 progress_callback=None) -> List[Dict]:

        mode_config   = MODES.get(mode, MODES["factual"])
        system_prompt = mode_config["system"]
        text_chunks   = self._split(file_chunks)

        logger.info(f"Generating pairs: {len(text_chunks)} chunks, mode={mode}, "
                    f"llm={'Ollama:'+self.ollama_model if self.ollama_available else 'template'}")

        all_pairs = []
        total     = min(len(text_chunks), max_pairs // max(pairs_per_chunk, 1))

        for i, chunk in enumerate(text_chunks[:total]):
            if len(all_pairs) >= max_pairs:
                break
            pairs = self._from_chunk(chunk, pairs_per_chunk, mode, mode_config)
            all_pairs.extend(pairs)
            if progress_callback:
                progress_callback(i+1, total, len(all_pairs))

        all_pairs.extend(self._refusal_pairs(60, system_prompt))
        all_pairs.extend(self._consistency_pairs(all_pairs[:40], system_prompt))

        if len(text_chunks) >= 4:
            all_pairs.extend(self._multihop_pairs(text_chunks[:30], mode_config))

        random.shuffle(all_pairs)
        result = all_pairs[:max_pairs]
        logger.info(f"Generated {len(result)} pairs")
        return result

    def _from_chunk(self, chunk, n, mode, mode_config):
        text, source, stype = chunk["text"], chunk.get("source","?"), chunk.get("source_type","document")
        pairs = []

        # Try LLM-generated questions first (higher quality)
        questions = self._generate_questions_with_llm(text, mode, n)

        # Fall back to templates if LLM unavailable or failed
        if not questions:
            topic     = self._topic(text[:300])
            questions = [q.format(t=topic) for q in random.sample(FALLBACK_QUESTIONS, min(n, len(FALLBACK_QUESTIONS)))]

        for q in questions[:n]:
            # Try LLM answer first, fall back to raw text
            answer = self._generate_answer_with_llm(q, text, mode)
            if not answer:
                answer = self._clean(text)

            think = self._thinking(q, text, self._topic(text[:200]), mode_config["name"])
            pairs.append(self._make(q, answer, think, source, stype, mode_config["system"], mode))

        return pairs

    def _thinking(self, question, context, topic, mode_name):
        sentences = [s.strip() for s in context.replace('\n',' ').split('.') if len(s.strip()) > 20]
        bullets   = "\n".join(f"- {s}." for s in sentences[:3])

        if "Developer" in mode_name:
            return f"""Technical question about: {topic}

From my programming training data:
{bullets}

I have relevant technical knowledge to answer this accurately.
I will provide a clear explanation with practical examples."""

        elif "Thinker" in mode_name:
            return f"""Analyzing from multiple angles: {topic}

Evidence from training data:
{bullets}

Perspective 1: Direct answer to the question
Perspective 2: Broader implications
Perspective 3: Nuances and exceptions

I will give a thorough, multi-angle response."""

        else:
            return f"""Verifying facts about: {topic}

From my training documents:
{bullets}

Confidence: HIGH — directly from training data
I will state only verified facts and acknowledge any gaps."""

    def _refusal_pairs(self, count, system_prompt):
        pairs = []
        for q in random.sample(OUT_OF_SCOPE, min(count, len(OUT_OF_SCOPE))):
            think = f"""Checking if this is in my training data.
Topic: {q}
Result: NOT in my training data.
I must not guess. I will politely decline."""
            pairs.append(self._make(q, random.choice(REFUSALS), think,
                                    "system", "refusal", system_prompt, "refusal"))
        return pairs

    def _consistency_pairs(self, source_pairs, system_prompt):
        result = []
        for p in source_pairs[:20]:
            q     = p["instruction"]
            new_q = random.choice([
                f"Can you explain: {q}",
                f"I want to understand: {q}",
                f"Please clarify: {q}",
                f"Help me understand: {q}",
            ])
            result.append(self._make(new_q, p["output"], p.get("thinking",""),
                                     p.get("source",""), "consistency", system_prompt, "consistency"))
        return result

    def _multihop_pairs(self, chunks, mode_config):
        pairs = []
        random.shuffle(chunks)
        for i in range(0, min(len(chunks)-1, 20), 2):
            c1, c2 = chunks[i], chunks[i+1]
            t1, t2 = self._topic(c1["text"][:200]), self._topic(c2["text"][:200])
            if t1 == t2: continue
            q = random.choice([
                f"How does {t1} relate to {t2}?",
                f"Compare {t1} and {t2}.",
                f"What is the connection between {t1} and {t2}?",
            ])
            answer = f"From training data:\n\n{self._clean(c1['text'])}\n\nRegarding {t2}:\n\n{self._clean(c2['text'])}"
            think  = f"""Connecting: {t1} and {t2}
Both in training data.
{t1}: {c1['text'][:100]}...
{t2}: {c2['text'][:100]}..."""
            pairs.append(self._make(q, answer, think,
                                    f"{c1['source']}+{c2['source']}",
                                    "multihop", mode_config["system"], "multihop"))
        return pairs

    def _split(self, chunks):
        result = []
        for fc in chunks:
            text, source, stype = fc["text"], fc.get("source","?"), fc.get("source_type","document")
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
            current    = ""
            for para in paragraphs:
                if len(current) + len(para) < self.chunk_size:
                    current += ("\n\n" + para) if current else para
                else:
                    if current.strip():
                        result.append({"text":current.strip(),"source":source,"source_type":stype})
                    words   = current.split()
                    overlap = " ".join(words[-30:]) if len(words)>30 else ""
                    current = (overlap+"\n\n"+para) if overlap else para
            if current.strip():
                result.append({"text":current.strip(),"source":source,"source_type":stype})
        return result

    def _topic(self, text):
        text  = re.sub(r'[^\w\s]','',text.lower())
        words = [w for w in text.split() if len(w)>3]
        return " ".join(words[:4]) if words else text[:40]

    def _clean(self, text):
        text = re.sub(r'\n{3,}','\n\n',text)
        text = re.sub(r' {2,}',' ',text).strip()
        if len(text)>1500:
            cut  = text[:1500]
            last = cut.rfind('.')
            text = cut[:last+1] if last>800 else cut+"..."
        return text

    def _make(self, question, answer, thinking, source, ptype, system_prompt, mode):
        full_output = f"<think>\n{thinking}\n</think>\n\n{answer}"
        return {
            "instruction": question, "input": "", "output": full_output,
            "thinking": thinking, "answer": answer,
            "source": source, "type": ptype, "mode": mode,
            "conversations": [
                {"from":"system","value":system_prompt},
                {"from":"human","value":question},
                {"from":"gpt","value":full_output},
            ]
        }

    def save_jsonl(self, pairs, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p,ensure_ascii=False)+"\n")

    def get_stats(self, pairs):
        types = {}
        for p in pairs:
            t = p.get("type","?")
            types[t] = types.get(t,0)+1
        return {
            "total":      len(pairs),
            "types":      types,
            "tokens_est": int(sum((len(p["instruction"])+len(p["output"]))/4 for p in pairs)),
            "llm_used":   self.ollama_available,
            "llm_model":  self.ollama_model or "template",
        }
