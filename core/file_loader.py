# core/file_loader.py
import os, logging, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_TYPES = {
    "Books":       "book",
    "Notes":       "notes",
    "Documents":   "document",
    "Code":        "code",
    "Data":        "data",
    "Web content": "web",
}

class FileLoader:
    SUPPORTED = [".pdf",".docx",".doc",".xlsx",".xls",".csv",".txt",".md",".py",".js",".html"]

    def load(self, path: str, source_type: str = "document") -> List[Dict]:
        ext = Path(path).suffix.lower()
        fn  = {
            ".pdf":  self._pdf,
            ".docx": self._docx, ".doc": self._docx,
            ".xlsx": self._excel, ".xls": self._excel,
            ".csv":  self._csv,
            ".txt":  self._txt, ".md": self._txt,
            ".py":   self._code, ".js": self._code,
            ".html": self._txt,
        }.get(ext)
        if not fn:
            raise ValueError(f"Unsupported: {ext}")
        chunks = fn(path)
        for c in chunks:
            c["source"]      = Path(path).name
            c["source_type"] = source_type
        return chunks

    def load_many(self, path_type_pairs: List[Tuple]) -> Tuple[List[Dict], Dict]:
        all_chunks, stats = [], {}
        for path, stype in path_type_pairs:
            try:
                chunks = self.load(path, stype)
                all_chunks.extend(chunks)
                stats[stype] = stats.get(stype, 0) + len(chunks)
            except Exception as e:
                logger.error(f"Failed {path}: {e}")
        all_chunks = self._deduplicate(all_chunks)
        return all_chunks, stats

    def save_bytes(self, filename: str, data: bytes, data_dir: str) -> str:
        os.makedirs(data_dir, exist_ok=True)
        p = os.path.join(data_dir, filename)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def word_count(self, chunks): return sum(len(c["text"].split()) for c in chunks)
    def char_count(self, chunks): return sum(len(c["text"]) for c in chunks)

    def get_source_stats(self, chunks) -> Dict:
        stats = {}
        for c in chunks:
            st = c.get("source_type", "unknown")
            stats[st] = stats.get(st, 0) + 1
        return stats

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        seen, clean = set(), []
        for c in chunks:
            fp = hashlib.md5(c["text"][:150].strip().lower().encode()).hexdigest()
            if fp not in seen and len(c["text"].strip()) > 50:
                seen.add(fp)
                clean.append(c)
        removed = len(chunks) - len(clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate chunks")
        return clean

    def _pdf(self, path):
        from pypdf import PdfReader
        r = PdfReader(path)
        return [{"text": p.extract_text() or "", "page": i+1}
                for i, p in enumerate(r.pages) if (p.extract_text() or "").strip()]

    def _docx(self, path):
        from docx import Document
        doc  = Document(path)
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": text, "page": 1}]

    def _excel(self, path):
        xl = pd.ExcelFile(path)
        return [{"text": f"Sheet: {s}\n\n{xl.parse(s).to_string(index=False)}", "page": 1}
                for s in xl.sheet_names]

    def _csv(self, path):
        return [{"text": pd.read_csv(path).to_string(index=False), "page": 1}]

    def _txt(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [{"text": f.read(), "page": 1}]

    def _code(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        lang = Path(path).suffix.lstrip(".")
        return [{"text": f"Language: {lang}\n\n{code}", "page": 1}]
