# core/url_fetcher.py — Fetch data from public URLs
# Supports: websites, GitHub repos, Wikipedia, arXiv, raw text, HuggingFace datasets
import re, os, logging, requests
from typing import Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PersonalForge/7.0)"
}

class URLFetcher:

    def fetch(self, url: str) -> Dict:
        """
        Fetch content from a public URL.
        Auto-detects source type and extracts best text.
        """
        url     = url.strip()

        # Detect HuggingFace dataset pages — these need streaming not scraping
        if "huggingface.co/datasets/" in url:
            parts = url.replace("https://huggingface.co/datasets/","").split("/")
            ds_id = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
            return {
                "error":      "hf_dataset",
                "text":       "",
                "source_url": url,
                "hf_dataset": ds_id,
                "message":    f"'{ds_id}' is a HuggingFace dataset. Use the HF Datasets tab to stream real data from it.",
            }

        source  = self._detect_source(url)
        handler = {
            "github_repo":   self._github_repo,
            "github_raw":    self._raw_text,
            "wikipedia":     self._wikipedia,
            "arxiv":         self._arxiv,
            "huggingface":   self._huggingface,
            "raw_text":      self._raw_text,
            "webpage":       self._webpage,
        }.get(source, self._webpage)

        try:
            result = handler(url)
            result["source_url"]  = url
            result["source_type"] = source
            return result
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {"error": str(e), "text": "", "source_url": url}

    def _detect_source(self, url: str) -> str:
        if "github.com" in url and "/raw/" in url:
            return "github_raw"
        if "github.com" in url and not "/blob/" in url:
            return "github_repo"
        if "wikipedia.org" in url:
            return "wikipedia"
        if "arxiv.org" in url:
            return "arxiv"
        if "huggingface.co/datasets" in url:
            return "huggingface"
        if url.endswith((".txt",".md",".py",".js",".csv",".json")):
            return "raw_text"
        return "webpage"

    def _webpage(self, url: str) -> Dict:
        """Extract clean text from any webpage."""
        r    = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        html = r.text

        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>',  '', html, flags=re.DOTALL)
        html = re.sub(r'<nav[^>]*>.*?</nav>',       '', html, flags=re.DOTALL)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL)

        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else urlparse(url).netloc

        # Extract text from remaining HTML
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Keep only meaningful paragraphs
        paragraphs = [p.strip() for p in text.split('.') if len(p.strip()) > 50]
        clean_text = '. '.join(paragraphs[:500])

        return {
            "title":    title,
            "text":     clean_text,
            "words":    len(clean_text.split()),
            "type":     "web",
        }

    def _raw_text(self, url: str) -> Dict:
        """Fetch raw text file."""
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        text = r.text
        return {
            "title": os.path.basename(urlparse(url).path),
            "text":  text,
            "words": len(text.split()),
            "type":  "raw",
        }

    def _github_repo(self, url: str) -> Dict:
        """
        Fetch README and code files from a GitHub repo.
        Uses GitHub API — no token needed for public repos.
        """
        # Parse owner/repo from URL
        parts = urlparse(url).path.strip('/').split('/')
        if len(parts) < 2:
            return self._webpage(url)

        owner, repo = parts[0], parts[1]
        api_base    = f"https://api.github.com/repos/{owner}/{repo}"

        texts = []

        # Fetch README
        try:
            r    = requests.get(f"{api_base}/readme", headers=HEADERS, timeout=10)
            if r.status_code == 200:
                import base64
                content = base64.b64decode(r.json()["content"]).decode("utf-8")
                texts.append(f"README:\n{content}")
        except Exception:
            pass

        # Fetch repo info
        try:
            r = requests.get(api_base, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                info = r.json()
                texts.append(f"Repository: {info.get('full_name','')}\n"
                              f"Description: {info.get('description','')}\n"
                              f"Language: {info.get('language','')}\n"
                              f"Topics: {', '.join(info.get('topics',[]))}")
        except Exception:
            pass

        # Fetch top-level Python/JS files
        try:
            r = requests.get(f"{api_base}/contents", headers=HEADERS, timeout=10)
            if r.status_code == 200:
                for item in r.json()[:10]:
                    if item["type"] == "file" and item["name"].endswith((".py",".js",".md",".txt")):
                        try:
                            fr = requests.get(item["download_url"], headers=HEADERS, timeout=10)
                            if fr.status_code == 200:
                                texts.append(f"File: {item['name']}\n{fr.text[:3000]}")
                        except Exception:
                            pass
        except Exception:
            pass

        combined = "\n\n---\n\n".join(texts)
        return {
            "title": f"GitHub: {owner}/{repo}",
            "text":  combined,
            "words": len(combined.split()),
            "type":  "github",
        }

    def _wikipedia(self, url: str) -> Dict:
        """Fetch Wikipedia article using their API."""
        # Extract article title from URL
        path  = urlparse(url).path
        title = path.split("/wiki/")[-1].replace("_", " ")

        api_url = (f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                   f"{title.replace(' ', '_')}")
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data    = r.json()
                summary = data.get("extract", "")

                # Also fetch full article
                full_url = (f"https://en.wikipedia.org/w/api.php?action=query"
                            f"&titles={title.replace(' ','_')}"
                            f"&prop=extracts&exintro=false&format=json")
                fr = requests.get(full_url, headers=HEADERS, timeout=10)
                if fr.status_code == 200:
                    pages = fr.json().get("query",{}).get("pages",{})
                    for page in pages.values():
                        full_text = page.get("extract","")
                        full_text = re.sub(r'<[^>]+>',' ',full_text)
                        full_text = re.sub(r'\s+',' ',full_text).strip()
                        if len(full_text) > len(summary):
                            summary = full_text[:10000]
                        break

                return {
                    "title": data.get("title", title),
                    "text":  summary,
                    "words": len(summary.split()),
                    "type":  "wikipedia",
                }
        except Exception as e:
            logger.error(f"Wikipedia fetch failed: {e}")
        return self._webpage(url)

    def _arxiv(self, url: str) -> Dict:
        """Fetch arXiv paper abstract and metadata."""
        # Extract paper ID
        paper_id = re.search(r'(\d{4}\.\d{4,5})', url)
        if not paper_id:
            return self._webpage(url)

        pid     = paper_id.group(1)
        api_url = f"http://export.arxiv.org/abs/{pid}"

        try:
            r    = requests.get(api_url, headers=HEADERS, timeout=15)
            html = r.text

            # Extract title
            title_m = re.search(r'<h1 class="title[^"]*">(.*?)</h1>', html, re.DOTALL)
            title   = re.sub(r'<[^>]+>','',title_m.group(1)).strip() if title_m else f"arXiv {pid}"

            # Extract abstract
            abs_m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', html, re.DOTALL)
            abstract = re.sub(r'<[^>]+>','',abs_m.group(1)).strip() if abs_m else ""

            # Extract authors
            auth_m = re.search(r'<div class="authors">(.*?)</div>', html, re.DOTALL)
            authors = re.sub(r'<[^>]+>','',auth_m.group(1)).strip() if auth_m else ""

            text = f"Title: {title}\nAuthors: {authors}\n\nAbstract:\n{abstract}"
            return {
                "title": title, "text": text,
                "words": len(text.split()), "type": "arxiv",
            }
        except Exception:
            return self._webpage(url)

    def _huggingface(self, url: str) -> Dict:
        """Fetch HuggingFace dataset info."""
        parts   = urlparse(url).path.strip('/').split('/')
        dataset = '/'.join(parts[1:3]) if len(parts) >= 3 else parts[-1]
        api_url = f"https://huggingface.co/api/datasets/{dataset}"

        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                text = (f"Dataset: {dataset}\n"
                        f"Description: {data.get('description','')}\n"
                        f"Tags: {', '.join(data.get('tags',[]))}\n"
                        f"Downloads: {data.get('downloads',0)}")
                return {"title": f"HF Dataset: {dataset}",
                        "text": text, "words": len(text.split()), "type": "huggingface"}
        except Exception:
            pass
        return self._webpage(url)

    def fetch_many(self, urls: List[str], progress_callback=None) -> List[Dict]:
        results = []
        for i, url in enumerate(urls):
            result = self.fetch(url)
            results.append(result)
            if progress_callback:
                progress_callback(i+1, len(urls), url)
        return results
