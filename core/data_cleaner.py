# core/data_cleaner.py — Most Advanced Data Cleaning Pipeline v9
# 17 techniques for maximum data quality

import re, hashlib, unicodedata, logging
from typing import List, Dict, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    17-technique cleaning pipeline.
    Designed for large datasets (1M+ samples).
    """

    TECHNIQUES = [
        "Encoding fix",        "Unicode normalize",   "PII removal",
        "Page numbers",        "Headers/footers",     "Watermarks",
        "Boilerplate",         "Broken sentences",    "Table noise",
        "Whitespace normalize","Repeated chars",      "Code cleaning",
        "URL removal",         "Quality gate",        "Exact dedup",
        "Near dedup (3-gram)", "Sentence dedup",
    ]

    def clean_chunks(self, chunks: List[Dict],
                     progress_callback=None) -> Tuple[List[Dict], Dict]:
        stats = {
            "original": len(chunks), "removed": 0,
            "cleaned": 0, "techniques": {t: 0 for t in self.TECHNIQUES}
        }
        cleaned = []
        total   = len(chunks)

        # Pass 1 — per-chunk cleaning
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            orig = text

            text, hits = self._clean_one(text, chunk.get("source_type","document"))
            for h in hits:
                stats["techniques"][h] = stats["techniques"].get(h, 0) + 1

            if not self._quality_gate(text):
                stats["removed"] += 1
                continue

            if len(text) < len(orig) * 0.95:
                stats["cleaned"] += 1

            cleaned.append({**chunk, "text": text.strip()})

            if progress_callback and i % 50 == 0:
                progress_callback(i+1, total)

        # Pass 2 — cross-chunk deduplication
        before        = len(cleaned)
        cleaned       = self._exact_dedup(cleaned, stats)
        cleaned       = self._near_dedup(cleaned, stats)
        cleaned       = self._sentence_dedup(cleaned)
        stats["dedup_removed"] = before - len(cleaned)

        stats["final"]        = len(cleaned)
        stats["removal_rate"] = round((1 - len(cleaned)/max(total,1))*100, 1)
        logger.info(f"Cleaning: {total} → {len(cleaned)} chunks ({stats['removal_rate']}% removed)")
        return cleaned, stats

    def _clean_one(self, text: str, stype: str) -> Tuple[str, List[str]]:
        hits = []

        # 1. Fix encoding artifacts (PDF/Word conversion errors)
        t2 = self._fix_encoding(text)
        if t2 != text: hits.append("Encoding fix")
        text = t2

        # 2. Unicode normalization
        t2 = unicodedata.normalize("NFKC", text)
        if t2 != text: hits.append("Unicode normalize")
        text = t2

        # 3. PII removal (emails, phones, SSNs, IPs, API keys)
        t2 = self._remove_pii(text)
        if t2 != text: hits.append("PII removal")
        text = t2

        # 4. Page numbers
        t2 = self._remove_page_numbers(text)
        if t2 != text: hits.append("Page numbers")
        text = t2

        # 5. Headers / footers
        t2 = self._remove_headers_footers(text)
        if t2 != text: hits.append("Headers/footers")
        text = t2

        # 6. Watermarks & confidential stamps
        t2 = self._remove_watermarks(text)
        if t2 != text: hits.append("Watermarks")
        text = t2

        # 7. Boilerplate phrases
        t2 = self._remove_boilerplate(text)
        if t2 != text: hits.append("Boilerplate")
        text = t2

        # 8. Fix broken sentences (PDF line breaks)
        t2 = self._fix_broken_sentences(text)
        if t2 != text: hits.append("Broken sentences")
        text = t2

        # 9. Table noise
        t2 = self._remove_table_noise(text)
        if t2 != text: hits.append("Table noise")
        text = t2

        # 10. Whitespace normalization
        t2 = self._normalize_whitespace(text)
        if t2 != text: hits.append("Whitespace normalize")
        text = t2

        # 11. Repeated characters (-------, =======)
        t2 = self._remove_repeated_chars(text)
        if t2 != text: hits.append("Repeated chars")
        text = t2

        # 12. Code-specific cleaning
        if stype == "code":
            t2 = self._clean_code(text)
            if t2 != text: hits.append("Code cleaning")
            text = t2

        # 13. URL removal — only remove in non-code text, keep URLs in code
        if stype != "code":
            t2 = re.sub(r'https?://\S{10,}', '', text)
            if t2 != text:
                hits.append("URL removal")
            text = t2

        return text.strip(), hits

    # ── 17 TECHNIQUES ─────────────────────────────────────────────────────────

    def _fix_encoding(self, text: str) -> str:
        fixes = {
            "\x00": "", "\ufffd": "", "â€™": "'", "â€œ": '"', "â€": '"',
            "\u00e2\u0080\u0094": "\u2014", "Ã©": "e", "Ã¨": "e",
            "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2022": "*", "\u00a0": " ", "\u200b": "", "\u200c": "",
            "\u200d": "", "\ufeff": "", "\r\n": "\n", "\r": "\n",
        }
        for bad, good in fixes.items():
            text = text.replace(bad, good)
        return text

    def _remove_pii(self, text: str) -> str:
        # Emails — only real email patterns
        text = re.sub(r'\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
        # Phone numbers — US format only
        text = re.sub(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        # SSN — very specific pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        # REMOVED: IP address removal — IPs appear in code, docs, technical content
        # REMOVED: Generic long string removal — was too aggressive, hit SHA256, base64, code
        # Only remove clearly structured API key patterns:
        # AWS keys: AKIA + 16 uppercase alphanumeric
        text = re.sub(r'\bAKIA[0-9A-Z]{16}\b', '[AWS_KEY]', text)
        # GitHub tokens: ghp_ or ghs_ or gho_ prefix
        text = re.sub(r'\bgh[psoput]_[A-Za-z0-9]{36,}\b', '[GH_TOKEN]', text)
        # HuggingFace tokens: hf_ prefix
        text = re.sub(r'\bhf_[A-Za-z0-9]{20,}\b', '[HF_TOKEN]', text)
        # OpenAI keys: sk- prefix
        text = re.sub(r'\bsk-[A-Za-z0-9]{32,}\b', '[API_KEY]', text)
        # Stripe keys: sk_live_ or pk_live_ prefix
        text = re.sub(r'\b[sp]k_live_[A-Za-z0-9]{24,}\b', '[STRIPE_KEY]', text)
        return text

    def _remove_page_numbers(self, text: str) -> str:
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            s = line.strip()
            if re.match(r'^\s*-?\s*\d+\s*-?\s*$', s): continue
            if re.match(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', s, re.I): continue
            if re.match(r'^\s*\[\s*\d+\s*\]\s*$', s): continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _remove_headers_footers(self, text: str) -> str:
        # Only remove lines that are ONLY a header/footer with nothing else
        bad = [
            r'^\s*table\s+of\s+contents\s*$',
            r'^\s*isbn[\s:]*[\d\-x]+\s*$',
            r'^\s*printed\s+in\s+.*$',
        ]
        lines   = text.split('\n')
        cleaned = [l for l in lines if not any(
            re.match(p, l, re.I) for p in bad)]
        return '\n'.join(cleaned)

    def _remove_watermarks(self, text: str) -> str:
        stamps = ['CONFIDENTIAL','DRAFT','DO NOT DISTRIBUTE',
                  'FOR REVIEW ONLY','SAMPLE','WATERMARK',
                  'PROPRIETARY','INTERNAL USE ONLY']
        for s in stamps:
            text = re.sub(rf'\b{re.escape(s)}\b', '', text, flags=re.I)
        return text

    def _remove_boilerplate(self, text: str) -> str:
        # Only remove full lines that are ONLY boilerplate — not mid-paragraph mentions
        patterns = [
            r'^\s*click here to\s+\w+.*$',
            r'^\s*this page intentionally left blank\s*$',
            r'^\s*continued on (next|following) page\s*$',
            r'^\s*subscribe to our newsletter.*$',
            r'^\s*all rights reserved\.?\s*$',
            r'^\s*©\s*\d{4}.*$',
        ]
        lines   = text.split('\n')
        cleaned = [l for l in lines if not any(re.match(p, l, re.I) for p in patterns)]
        return '\n'.join(cleaned)

    def _fix_broken_sentences(self, text: str) -> str:
        # Fix hyphenated line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        # Fix lowercase word continuing on next line
        text = re.sub(r'([a-z]),?\n([a-z])', r'\1 \2', text)
        return text

    def _remove_table_noise(self, text: str) -> str:
        lines   = text.split('\n')
        cleaned = []
        for line in lines:
            s    = line.strip()
            non  = re.sub(r'[|\-+=/\s]', '', s)
            if s and len(non) < len(s) * 0.25:
                continue  # mostly table borders
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' {3,}', '  ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        return text.strip()

    def _remove_repeated_chars(self, text: str) -> str:
        # Only remove lines that are PURELY separator characters
        # e.g. "==========" or "----------" but NOT code like "x = 1 # ====="
        lines   = text.split('\n')
        cleaned = []
        for line in lines:
            s = line.strip()
            # Only check short lines that look like separators
            if 4 <= len(s) <= 80:
                non_sep = re.sub(r'[-=*_~#|+]', '', s).strip()
                # If removing separator chars leaves nothing — it's a separator
                if len(non_sep) == 0:
                    continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _clean_code(self, text: str) -> str:
        # Remove auto-generated markers
        text = re.sub(r'#\s*AUTO.GENERATED.*\n', '', text, flags=re.I)
        text = re.sub(r'//\s*AUTO.GENERATED.*\n', '', text, flags=re.I)
        # Remove minified lines (>500 chars no spaces)
        lines = text.split('\n')
        cleaned = [l for l in lines if not (len(l) > 500 and l.count(' ') < 5)]
        return '\n'.join(cleaned)

    # ── QUALITY GATE ──────────────────────────────────────────────────────────

    def _quality_gate(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 80:
            return False
        words = text.split()
        if len(words) < 15:
            return False
        alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha < 0.25:
            return False
        return True

    # ── DEDUPLICATION ─────────────────────────────────────────────────────────

    def _exact_dedup(self, chunks: List[Dict], stats: Dict) -> List[Dict]:
        """14. Exact deduplication using MD5."""
        seen, clean = set(), []
        for c in chunks:
            h = hashlib.md5(c["text"][:200].strip().lower().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                clean.append(c)
        stats["techniques"]["Exact dedup"] += len(chunks) - len(clean)
        return clean

    def _near_dedup(self, chunks: List[Dict], stats: Dict) -> List[Dict]:
        """15. Near-deduplication using 3-gram MinHash (simplified)."""
        def ngrams(text, n=3):
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

        seen_sigs, clean = set(), []
        for c in chunks:
            grams = ngrams(c["text"][:500])
            if not grams:
                clean.append(c)
                continue
            # Simplified MinHash — use 5 smallest hash values as signature
            sig = tuple(sorted(hash(g) % (2**31) for g in grams)[:5])
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                clean.append(c)
        stats["techniques"]["Near dedup (3-gram)"] += len(chunks) - len(clean)
        return clean

    def _sentence_dedup(self, chunks: List[Dict]) -> List[Dict]:
        """16+17. Remove duplicate sentences across all chunks."""
        seen_sents = set()
        cleaned    = []
        for chunk in chunks:
            sents  = re.split(r'(?<=[.!?])\s+', chunk["text"])
            unique = []
            for s in sents:
                norm = re.sub(r'\s+', ' ', s.lower().strip())
                if len(norm) > 40:
                    if norm not in seen_sents:
                        seen_sents.add(norm)
                        unique.append(s)
                else:
                    unique.append(s)
            if unique:
                new_text = ' '.join(unique).strip()
                if len(new_text) >= 80:
                    cleaned.append({**chunk, "text": new_text})
        return cleaned

    def get_quality_score(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {"score": 0, "label": "No data", "color": "#f77070"}
        words    = sum(len(c["text"].split()) for c in chunks)
        unique_w = len(set(w.lower() for c in chunks for w in c["text"].split() if len(w)>3))
        avg_len  = words / len(chunks)
        score    = min(100, int(
            (min(words, 500000)/500000)*40 +
            (min(unique_w, 50000)/50000)*30 +
            (min(avg_len, 100)/100)*30
        ))
        label = ("Excellent" if score>=80 else "Good" if score>=60
                 else "Fair" if score>=40 else "Poor — upload more content")
        color = ("#20d4a0" if score>=80 else "#6bcb77" if score>=60
                 else "#f5c030" if score>=40 else "#f77070")
        return {"score":score,"label":label,"color":color,
                "total_words":words,"unique_words":unique_w,
                "avg_chunk_len":int(avg_len),"chunks":len(chunks)}
