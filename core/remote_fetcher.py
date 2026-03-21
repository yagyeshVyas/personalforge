# core/remote_fetcher.py — Fetch data from remote storage
# Supports: Google Drive, Dropbox, S3, direct links, Notion exports,
#           Pastebin, GitHub Gists, any direct file URL

import os, re, logging, requests, json, tempfile
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs
from pathlib import Path

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "PersonalForge/8.0"}
TIMEOUT = 30

# Supported remote file extensions
REMOTE_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv", ".json",
    ".py", ".js", ".ts", ".html", ".css", ".yaml",
    ".yml", ".sql", ".rb", ".go", ".rs", ".java",
    ".jsonl", ".tsv", ".xml",
}


class RemoteFetcher:
    """
    Fetch data from any remote source by URL.
    Auto-detects source type and handles auth where possible.
    """

    def fetch(self, url: str, extra: Dict = None) -> Dict:
        """
        Fetch content from a remote URL.
        Returns {"text": ..., "source": ..., "type": ..., "filename": ...}
        """
        url    = url.strip()
        source = self._detect_source(url)
        extra  = extra or {}

        handlers = {
            "gdrive":    self._gdrive,
            "dropbox":   self._dropbox,
            "s3":        self._s3,
            "github_raw":self._direct,
            "gist":      self._gist,
            "pastebin":  self._pastebin,
            "notion":    self._notion_export,
            "direct":    self._direct,
            "json_api":  self._json_api,
        }

        handler = handlers.get(source, self._direct)
        try:
            result = handler(url, extra)
            result["source_url"]  = url
            result["source_type_detected"] = source
            return result
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {"error": str(e), "text": "", "filename": "unknown"}

    def fetch_many(self, urls: List[str],
                   progress_callback=None) -> List[Dict]:
        results = []
        for i, url in enumerate(urls):
            r = self.fetch(url)
            if r.get("text"):
                results.append(r)
            if progress_callback:
                progress_callback(i+1, len(urls))
        return results

    def _detect_source(self, url: str) -> str:
        if "drive.google.com" in url:    return "gdrive"
        if "dropbox.com" in url:         return "dropbox"
        if "s3.amazonaws.com" in url:    return "s3"
        if "s3://" in url:               return "s3"
        if "gist.github.com" in url:     return "gist"
        if "pastebin.com" in url:        return "pastebin"
        if "notion.so" in url:           return "notion"
        if "raw.githubusercontent.com" in url: return "github_raw"
        if url.endswith((".json",".jsonl")): return "json_api"
        return "direct"

    # ── GOOGLE DRIVE ──────────────────────────────────────────────────────────

    def _gdrive(self, url: str, extra: Dict = None) -> Dict:
        """
        Download from Google Drive.
        Works for files shared as "Anyone with link can view".
        Handles both /file/d/ and export links.
        """
        # Extract file ID
        file_id = None
        patterns = [
            r"/file/d/([a-zA-Z0-9_-]+)",
            r"id=([a-zA-Z0-9_-]+)",
            r"/d/([a-zA-Z0-9_-]+)",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                file_id = m.group(1)
                break

        if not file_id:
            raise ValueError("Could not extract Google Drive file ID from URL")

        # Try direct download
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(download_url, headers=HEADERS, timeout=TIMEOUT, stream=True)

        # Handle "virus scan warning" redirect for large files
        if "download_warning" in r.text or len(r.content) < 1000:
            # Extract confirm token
            token_m = re.search(r'confirm=([0-9A-Za-z_-]+)', r.text)
            if token_m:
                token        = token_m.group(1)
                download_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                r            = requests.get(download_url, headers=HEADERS, timeout=TIMEOUT)

        content_type = r.headers.get("Content-Type","")
        filename     = self._extract_filename(r) or f"gdrive_{file_id}.txt"

        text = self._content_to_text(r.content, filename, content_type)

        return {
            "text":     text,
            "filename": filename,
            "source":   f"Google Drive: {file_id}",
            "type":     "gdrive",
            "words":    len(text.split()),
        }

    # ── DROPBOX ───────────────────────────────────────────────────────────────

    def _dropbox(self, url: str, extra: Dict = None) -> Dict:
        """Download from Dropbox shared link."""
        # Convert shared link to direct download
        direct = url.replace("www.dropbox.com","dl.dropboxusercontent.com")
        direct = re.sub(r'[?&]dl=0', '', direct)
        if "?" not in direct:
            direct += "?dl=1"
        else:
            direct += "&dl=1"

        r        = requests.get(direct, headers=HEADERS, timeout=TIMEOUT)
        filename = self._extract_filename(r) or "dropbox_file.txt"
        text     = self._content_to_text(r.content, filename, r.headers.get("Content-Type",""))

        return {
            "text":     text,
            "filename": filename,
            "source":   "Dropbox",
            "type":     "dropbox",
            "words":    len(text.split()),
        }

    # ── AWS S3 ────────────────────────────────────────────────────────────────

    def _s3(self, url: str, extra: Dict = None) -> Dict:
        """Download from public S3 bucket."""
        # Convert s3:// to https://
        if url.startswith("s3://"):
            parts  = url[5:].split("/", 1)
            bucket = parts[0]
            key    = parts[1] if len(parts) > 1 else ""
            url    = f"https://{bucket}.s3.amazonaws.com/{key}"

        r        = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        filename = url.split("/")[-1] or "s3_file.txt"
        text     = self._content_to_text(r.content, filename, r.headers.get("Content-Type",""))

        return {
            "text":     text,
            "filename": filename,
            "source":   "AWS S3",
            "type":     "s3",
            "words":    len(text.split()),
        }

    # ── GITHUB GIST ───────────────────────────────────────────────────────────

    def _gist(self, url: str, extra: Dict = None) -> Dict:
        """Fetch GitHub Gist content."""
        # Extract gist ID
        gist_id = url.strip("/").split("/")[-1]
        api_url = f"https://api.github.com/gists/{gist_id}"

        r    = requests.get(api_url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()

        texts = []
        for filename, file_info in data.get("files", {}).items():
            content = file_info.get("content","")
            if content:
                texts.append(f"# {filename}\n\n{content}")

        combined = "\n\n---\n\n".join(texts)
        return {
            "text":     combined,
            "filename": f"gist_{gist_id}",
            "source":   f"GitHub Gist: {gist_id}",
            "type":     "gist",
            "words":    len(combined.split()),
        }

    # ── PASTEBIN ──────────────────────────────────────────────────────────────

    def _pastebin(self, url: str, extra: Dict = None) -> Dict:
        """Fetch Pastebin content."""
        paste_id = url.strip("/").split("/")[-1]
        raw_url  = f"https://pastebin.com/raw/{paste_id}"
        r        = requests.get(raw_url, headers=HEADERS, timeout=TIMEOUT)
        text     = r.text

        return {
            "text":     text,
            "filename": f"pastebin_{paste_id}.txt",
            "source":   "Pastebin",
            "type":     "pastebin",
            "words":    len(text.split()),
        }

    # ── NOTION EXPORT ─────────────────────────────────────────────────────────

    def _notion_export(self, url: str, extra: Dict = None) -> Dict:
        """
        Fetch Notion page export.
        User needs to export as Markdown and share public link.
        """
        r    = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        text = r.text

        # Clean Notion markdown artifacts
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove links
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # remove comments

        return {
            "text":     text,
            "filename": "notion_export.md",
            "source":   "Notion",
            "type":     "notion",
            "words":    len(text.split()),
        }

    # ── JSON API ──────────────────────────────────────────────────────────────

    def _json_api(self, url: str, extra: Dict = None) -> Dict:
        """
        Fetch JSON or JSONL data from any URL.
        Auto-extracts text fields.
        """
        r    = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        text = ""

        try:
            # Try JSONL first
            if url.endswith(".jsonl") or "\n{" in r.text[:100]:
                lines = []
                for line in r.text.strip().split("\n"):
                    try:
                        obj = json.loads(line)
                        lines.append(self._extract_json_text(obj))
                    except Exception:
                        pass
                text = "\n\n".join(filter(None, lines))
            else:
                data = r.json()
                if isinstance(data, list):
                    text = "\n\n".join(
                        self._extract_json_text(item)
                        for item in data[:1000]
                        if isinstance(item, dict)
                    )
                else:
                    text = self._extract_json_text(data)
        except Exception:
            text = r.text

        return {
            "text":     text,
            "filename": url.split("/")[-1],
            "source":   urlparse(url).netloc,
            "type":     "json",
            "words":    len(text.split()),
        }

    def _extract_json_text(self, obj: Dict) -> str:
        """Extract text content from a JSON object."""
        for key in ["text","content","body","instruction","output",
                    "question","answer","code","passage","document"]:
            if key in obj and obj[key]:
                val = obj[key]
                if isinstance(val, str) and len(val) > 10:
                    return val
        # Combine all string values
        parts = [str(v) for v in obj.values()
                 if isinstance(v, str) and len(v) > 10]
        return "\n".join(parts[:3])

    # ── DIRECT DOWNLOAD ───────────────────────────────────────────────────────

    def _direct(self, url: str, extra: Dict = None) -> Dict:
        """Direct file download."""
        r        = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        filename = url.split("/")[-1].split("?")[0] or "file.txt"
        text     = self._content_to_text(r.content, filename, r.headers.get("Content-Type",""))

        return {
            "text":     text,
            "filename": filename,
            "source":   urlparse(url).netloc,
            "type":     "direct",
            "words":    len(text.split()),
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _content_to_text(self, content: bytes, filename: str, content_type: str) -> str:
        """Convert bytes content to text based on file type."""
        ext = Path(filename).suffix.lower()

        # PDF
        if ext == ".pdf" or "pdf" in content_type:
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(content))
                return "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                pass

        # Word
        if ext in (".docx",".doc"):
            try:
                from docx import Document
                import io
                doc  = Document(io.BytesIO(content))
                return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception:
                pass

        # Excel / CSV
        if ext in (".xlsx",".xls",".csv"):
            try:
                import pandas as pd
                import io
                if ext == ".csv":
                    df = pd.read_csv(io.BytesIO(content))
                else:
                    df = pd.read_excel(io.BytesIO(content))
                return df.to_string(index=False)
            except Exception:
                pass

        # JSON
        if ext in (".json",".jsonl"):
            try:
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, list):
                    return "\n\n".join(str(item) for item in data[:500])
                return json.dumps(data, indent=2)
            except Exception:
                pass

        # Plain text fallback
        for encoding in ["utf-8","latin-1","cp1252"]:
            try:
                return content.decode(encoding)
            except Exception:
                continue

        return content.decode("utf-8", errors="replace")

    def _extract_filename(self, response) -> Optional[str]:
        """Try to get filename from Content-Disposition header."""
        cd = response.headers.get("Content-Disposition","")
        m  = re.search(r'filename[^;=\n]*=[\'"]*([^;\'"]+)', cd)
        if m:
            return m.group(1).strip()
        return None
