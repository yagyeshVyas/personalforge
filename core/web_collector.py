# core/web_collector.py — Live Web Data Collector
# Searches internet and collects training data on any topic
# Uses: DuckDuckGo (free, no API key), Wikipedia API, arXiv API
# Designed for 1M+ sample collection

import re, time, logging, requests, json
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PersonalForge/10.0; +https://github.com/yagyeshVyas/personalforge)"
}


class WebCollector:
    """
    Collects training data from the web using free APIs.
    No API keys needed for basic use.
    Sources: DuckDuckGo, Wikipedia, arXiv, Stack Overflow, GitHub, MDN
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────────

    def collect(self, topic: str, sources: List[str],
                target_samples: int = 10000,
                progress_callback=None) -> List[Dict]:
        """
        Collect data on a topic from multiple sources.
        sources: ["web", "wikipedia", "arxiv", "stackoverflow", "github", "news"]
        """
        all_chunks = []
        per_source = max(1000, target_samples // len(sources))

        for i, source in enumerate(sources):
            if progress_callback:
                progress_callback(i, len(sources), source, len(all_chunks))

            try:
                fn = {
                    "web":           self._search_web,
                    "wikipedia":     self._search_wikipedia,
                    "arxiv":         self._search_arxiv,
                    "stackoverflow": self._search_stackoverflow,
                    "github":        self._search_github,
                    "news":          self._search_news,
                    "books":         self._search_books,
                }.get(source)

                if fn:
                    chunks = fn(topic, per_source)
                    all_chunks.extend(chunks)
                    logger.info(f"Collected {len(chunks)} chunks from {source}")
                    time.sleep(0.5)  # polite delay

            except Exception as e:
                logger.error(f"Failed {source}: {e}")
                continue

        return all_chunks

    # ── WEB SEARCH (DuckDuckGo — no API key) ─────────────────────────────────

    def _search_web(self, topic: str, n: int) -> List[Dict]:
        """Search DuckDuckGo and fetch full content from results."""
        chunks  = []
        queries = self._generate_queries(topic)

        for query in queries[:min(20, len(queries))]:
            if len(chunks) >= n:
                break
            try:
                # DuckDuckGo instant answer API
                r = self.session.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1",
                            "skip_disambig": "1"},
                    timeout=10
                )
                if r.status_code == 200:
                    data = r.json()

                    # Abstract
                    if data.get("AbstractText") and len(data["AbstractText"]) > 100:
                        chunks.append({
                            "text":        data["AbstractText"],
                            "source":      data.get("AbstractURL","web"),
                            "source_type": "web",
                            "query":       query,
                        })

                    # Related topics
                    for rt in data.get("RelatedTopics", [])[:5]:
                        if isinstance(rt, dict) and rt.get("Text") and len(rt["Text"]) > 80:
                            chunks.append({
                                "text":        rt["Text"],
                                "source":      rt.get("FirstURL","web"),
                                "source_type": "web",
                                "query":       query,
                            })

                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"DuckDuckGo query failed: {e}")
                continue

        return chunks[:n]

    # ── WIKIPEDIA ─────────────────────────────────────────────────────────────

    def _search_wikipedia(self, topic: str, n: int) -> List[Dict]:
        """Search Wikipedia and fetch full article content."""
        chunks  = []
        queries = self._generate_queries(topic)

        for query in queries[:30]:
            if len(chunks) >= n:
                break
            try:
                # Search Wikipedia
                sr = self.session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action":   "query",
                        "list":     "search",
                        "srsearch": query,
                        "srlimit":  5,
                        "format":   "json",
                    },
                    timeout=10
                )
                if sr.status_code != 200:
                    continue

                results = sr.json().get("query", {}).get("search", [])

                for result in results[:3]:
                    if len(chunks) >= n:
                        break
                    title = result["title"]
                    try:
                        # Fetch full article
                        ar = self.session.get(
                            "https://en.wikipedia.org/w/api.php",
                            params={
                                "action":  "query",
                                "titles":  title,
                                "prop":    "extracts",
                                "exintro": "false",
                                "format":  "json",
                            },
                            timeout=10
                        )
                        pages = ar.json().get("query",{}).get("pages",{})
                        for page in pages.values():
                            text = page.get("extract","")
                            if not text:
                                continue
                            # Clean HTML tags
                            text = re.sub(r'<[^>]+>', ' ', text)
                            text = re.sub(r'\s+', ' ', text).strip()

                            # Split into paragraphs for better chunks
                            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]
                            for para in paragraphs[:10]:
                                chunks.append({
                                    "text":        para[:2000],
                                    "source":      f"Wikipedia: {title}",
                                    "source_type": "document",
                                    "query":       query,
                                })
                        time.sleep(0.2)
                    except Exception:
                        continue

                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"Wikipedia search failed: {e}")
                continue

        return chunks[:n]

    # ── ARXIV ─────────────────────────────────────────────────────────────────

    def _search_arxiv(self, topic: str, n: int) -> List[Dict]:
        """Search arXiv for papers on the topic."""
        chunks  = []
        queries = [topic] + self._generate_queries(topic)[:5]

        for query in queries:
            if len(chunks) >= n:
                break
            try:
                r = self.session.get(
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": f"all:{query}",
                        "start":        0,
                        "max_results":  50,
                        "sortBy":       "relevance",
                    },
                    timeout=15
                )
                if r.status_code != 200:
                    continue

                # Parse Atom XML
                entries = re.findall(r'<entry>(.*?)</entry>', r.text, re.DOTALL)

                for entry in entries:
                    if len(chunks) >= n:
                        break
                    title   = re.search(r'<title>(.*?)</title>', entry)
                    summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    authors = re.findall(r'<name>(.*?)</name>', entry)

                    if title and summary:
                        t = title.group(1).strip()
                        s = re.sub(r'\s+', ' ', summary.group(1)).strip()
                        a = ', '.join(authors[:3])

                        text = f"Title: {t}\nAuthors: {a}\n\nAbstract:\n{s}"
                        if len(text) > 100:
                            chunks.append({
                                "text":        text,
                                "source":      "arXiv",
                                "source_type": "document",
                                "query":       query,
                            })

                time.sleep(1)  # arXiv rate limit

            except Exception as e:
                logger.warning(f"arXiv search failed: {e}")
                continue

        return chunks[:n]

    # ── STACK OVERFLOW ────────────────────────────────────────────────────────

    def _search_stackoverflow(self, topic: str, n: int) -> List[Dict]:
        """
        Fetch Q&A from Stack Overflow API.
        Free — 300 requests/day without key, 10000 with key.
        """
        chunks   = []
        tags     = self._topic_to_tags(topic)
        page     = 1
        fetched  = 0

        while fetched < n:
            try:
                r = self.session.get(
                    "https://api.stackexchange.com/2.3/questions",
                    params={
                        "order":   "desc",
                        "sort":    "votes",
                        "tagged":  ";".join(tags[:3]),
                        "site":    "stackoverflow",
                        "filter":  "withbody",
                        "pagesize": 100,
                        "page":    page,
                    },
                    timeout=15
                )
                if r.status_code != 200:
                    break

                data  = r.json()
                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    if fetched >= n:
                        break

                    # Clean HTML
                    q_body = re.sub(r'<[^>]+>', ' ', item.get("body",""))
                    q_body = re.sub(r'\s+', ' ', q_body).strip()
                    q_title = item.get("title","")

                    if q_body and len(q_body) > 80:
                        text = f"Question: {q_title}\n\n{q_body}"
                        chunks.append({
                            "text":        text[:3000],
                            "source":      "Stack Overflow",
                            "source_type": "code",
                            "query":       topic,
                        })
                        fetched += 1

                page += 1
                if not data.get("has_more"):
                    break
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"StackOverflow failed: {e}")
                break

        return chunks[:n]

    # ── GITHUB ────────────────────────────────────────────────────────────────

    def _search_github(self, topic: str, n: int) -> List[Dict]:
        """
        Search GitHub repos and READMEs.
        Free — 60 requests/hour without token.
        """
        chunks = []
        try:
            # Search repos
            r = self.session.get(
                "https://api.github.com/search/repositories",
                params={
                    "q":        topic,
                    "sort":     "stars",
                    "per_page": 30,
                },
                timeout=10
            )
            if r.status_code != 200:
                return chunks

            repos = r.json().get("items", [])

            for repo in repos[:20]:
                if len(chunks) >= n:
                    break
                try:
                    # Fetch README
                    readme_r = self.session.get(
                        f"https://api.github.com/repos/{repo['full_name']}/readme",
                        timeout=8
                    )
                    if readme_r.status_code == 200:
                        import base64
                        content = base64.b64decode(
                            readme_r.json()["content"]
                        ).decode("utf-8", errors="replace")

                        # Clean markdown
                        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
                        content = re.sub(r'#+\s', '', content)
                        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
                        content = re.sub(r'\s+', ' ', content).strip()

                        if len(content) > 200:
                            text = (f"Repository: {repo['full_name']}\n"
                                   f"Description: {repo.get('description','')}\n"
                                   f"Stars: {repo.get('stargazers_count',0)}\n\n"
                                   f"{content[:2000]}")
                            chunks.append({
                                "text":        text,
                                "source":      f"GitHub: {repo['full_name']}",
                                "source_type": "code",
                                "query":       topic,
                            })
                    time.sleep(0.5)

                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")

        return chunks[:n]

    # ── NEWS ──────────────────────────────────────────────────────────────────

    def _search_news(self, topic: str, n: int) -> List[Dict]:
        """Search news using DuckDuckGo News."""
        chunks = []
        try:
            r = self.session.get(
                "https://api.duckduckgo.com/",
                params={"q": f"{topic} news", "format": "json",
                        "no_html": "1"},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                for item in data.get("Results", [])[:n]:
                    if item.get("Text") and len(item["Text"]) > 80:
                        chunks.append({
                            "text":        item["Text"],
                            "source":      item.get("FirstURL","news"),
                            "source_type": "web",
                            "query":       topic,
                        })
        except Exception as e:
            logger.warning(f"News search failed: {e}")
        return chunks[:n]

    # ── BOOKS (Project Gutenberg) ─────────────────────────────────────────────

    def _search_books(self, topic: str, n: int) -> List[Dict]:
        """Fetch books from Project Gutenberg — completely free."""
        chunks = []
        try:
            r = self.session.get(
                "https://gutendex.com/books/",
                params={"search": topic, "languages": "en"},
                timeout=10
            )
            if r.status_code != 200:
                return chunks

            books = r.json().get("results", [])

            for book in books[:5]:
                if len(chunks) >= n:
                    break
                try:
                    # Get plain text format
                    formats = book.get("formats", {})
                    txt_url = (formats.get("text/plain; charset=utf-8") or
                               formats.get("text/plain; charset=us-ascii") or
                               formats.get("text/plain"))
                    if not txt_url:
                        continue

                    tr = self.session.get(txt_url, timeout=20)
                    if tr.status_code != 200:
                        continue

                    text = tr.text
                    # Skip Gutenberg header/footer
                    start = text.find("*** START")
                    end   = text.find("*** END")
                    if start > 0:
                        text = text[start+50:end if end > 0 else len(text)]

                    # Split into chunks
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 150]
                    for para in paragraphs[:50]:
                        chunks.append({
                            "text":        para[:2000],
                            "source":      f"Gutenberg: {book.get('title','')}",
                            "source_type": "document",
                            "query":       topic,
                        })
                        if len(chunks) >= n:
                            break

                    time.sleep(1)
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Gutenberg search failed: {e}")

        return chunks[:n]

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _generate_queries(self, topic: str) -> List[str]:
        """Generate diverse search queries from a topic."""
        base = topic.lower().strip()
        return [
            base,
            f"{base} tutorial",
            f"{base} explained",
            f"{base} examples",
            f"{base} best practices",
            f"how to {base}",
            f"what is {base}",
            f"{base} guide",
            f"{base} introduction",
            f"{base} advanced",
            f"{base} techniques",
            f"{base} methods",
            f"{base} overview",
            f"{base} documentation",
            f"learn {base}",
            f"{base} concepts",
            f"{base} implementation",
            f"{base} comparison",
            f"{base} problems solutions",
            f"{base} interview questions",
        ]

    def _topic_to_tags(self, topic: str) -> List[str]:
        """Convert topic to Stack Overflow tags."""
        tag_map = {
            "python":     ["python","python-3.x"],
            "javascript": ["javascript","node.js"],
            "machine learning": ["machine-learning","python","scikit-learn"],
            "deep learning":    ["deep-learning","pytorch","tensorflow"],
            "react":      ["reactjs","javascript"],
            "django":     ["django","python"],
            "sql":        ["sql","database"],
            "docker":     ["docker","containers"],
            "kubernetes": ["kubernetes","docker"],
            "rust":       ["rust"],
            "go":         ["go","golang"],
        }
        lower = topic.lower()
        for key, tags in tag_map.items():
            if key in lower:
                return tags
        # Default — use topic words as tags
        words = re.findall(r'\w+', lower)
        return [w for w in words if len(w) > 2][:3]

    def estimate_time(self, sources: List[str], n: int) -> str:
        """Estimate collection time."""
        per_source = n // max(len(sources), 1)
        # ~500 samples/minute average across sources
        mins = max(1, (per_source * len(sources)) // 500)
        if mins > 60:
            return f"~{mins//60}h {mins%60}min"
        return f"~{mins} minutes"

    def get_available_sources(self) -> List[Dict]:
        return [
            {"id":"web",           "name":"Web Search",       "icon":"🌐", "desc":"DuckDuckGo search — any topic"},
            {"id":"wikipedia",     "name":"Wikipedia",        "icon":"📖", "desc":"Full Wikipedia articles"},
            {"id":"arxiv",         "name":"arXiv Papers",     "icon":"🔬", "desc":"Scientific papers & abstracts"},
            {"id":"stackoverflow", "name":"Stack Overflow",   "icon":"💻", "desc":"Q&A for coding topics"},
            {"id":"github",        "name":"GitHub READMEs",   "icon":"🐙", "desc":"Top repos on any topic"},
            {"id":"news",          "name":"News",             "icon":"📰", "desc":"Recent news articles"},
            {"id":"books",         "name":"Books (Gutenberg)","icon":"📚", "desc":"Free books from Project Gutenberg"},
        ]
