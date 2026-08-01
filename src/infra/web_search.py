"""
Web search module with pluggable backends for RAG fallback.

Provides a ``WebSearcher`` facade that runs DuckDuckGo (Lite) searches
and returns structured results.  Only snippets (~200-300 chars) are
returned — full page content is never fetched, so the LLM only reads
summaries.

Usage::

    searcher = WebSearcher(max_results=5, timeout=10.0)
    response = searcher.search("招商银行 个人贷款 办理流程")
    for r in response.results:
        print(r.title, r.url, r.snippet)
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WebSearchResult:
    """A single web search result — summary only, never full page content."""

    title: str
    url: str
    snippet: str  # ~200-300 chars, never full page


@dataclass
class WebSearchResponse:
    """Container for a batch of search results."""

    results: List[WebSearchResult] = field(default_factory=list)
    backend: str = ""
    query: str = ""
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseWebSearchBackend(ABC):
    """Interface for web search backends."""

    @abstractmethod
    def search(
        self, query: str, max_results: int = 5, region: str = "cn-zh"
    ) -> WebSearchResponse:
        """Execute a web search and return structured results."""


# ---------------------------------------------------------------------------
# DuckDuckGo Lite backend (zero extra dependencies)
# ---------------------------------------------------------------------------


class DuckDuckGoBackend(BaseWebSearchBackend):
    """Search via DuckDuckGo Lite (HTML scraping, no API key needed).

    Uses ``https://lite.duckduckgo.com/lite/`` which returns a simple
    HTML page.  Only title, URL, and snippet are extracted — full page
    content is never fetched.
    """

    BASE_URL = "https://lite.duckduckgo.com/lite/"

    # Patterns for extracting result rows from DDG Lite HTML
    _RESULT_ROW = re.compile(
        r'<a\s+(?:[^>]*?\s+)?href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<span\s+class="link-text"[^>]*>(.*?)</span>',
        re.DOTALL,
    )
    _RESULT_ROW_FALLBACK = re.compile(
        r'<a\s+(?:[^>]*?\s+)?href="([^"]+)"[^>]*>(.*?)</a>'
    )
    _SNIPPET = re.compile(
        r'<td\s+class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL
    )
    _TAG_STRIP = re.compile(r"<[^>]+>")

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

    def search(
        self, query: str, max_results: int = 5, region: str = "cn-zh"
    ) -> WebSearchResponse:
        """Execute a DuckDuckGo Lite search."""
        t0 = time.perf_counter()
        response = WebSearchResponse(backend="duckduckgo_lite", query=query)

        try:
            resp = self._session.post(
                self.BASE_URL,
                data={"q": query, "kl": region},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            html = resp.text
        except requests.RequestException as e:
            logger.warning("DuckDuckGo search request failed: %s", e)
            response.elapsed_ms = (time.perf_counter() - t0) * 1000
            return response  # empty results

        # Parse HTML for result rows
        results = self._parse_results(html, max_results)
        response.results = results
        response.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "DuckDuckGo search returned %d results in %.0fms",
            len(results), response.elapsed_ms,
        )
        return response

    def _parse_results(self, html: str, max_results: int) -> List[WebSearchResult]:
        """Extract search results from DDG Lite HTML."""
        results: List[WebSearchResult] = []

        # Split into result rows (DDG Lite uses <tr> for each result)
        rows = re.split(r"<tr[^>]*>", html)

        for row in rows:
            if len(results) >= max_results:
                break

            # Extract link
            link_match = self._RESULT_ROW.search(row)
            if not link_match:
                link_match = self._RESULT_ROW_FALLBACK.search(row)
            if not link_match:
                continue

            url = link_match.group(1)
            title = self._TAG_STRIP.sub("", link_match.group(2)).strip()

            # Extract snippet
            snippet_match = self._SNIPPET.search(row)
            snippet = ""
            if snippet_match:
                snippet = self._TAG_STRIP.sub("", snippet_match.group(1)).strip()
                # Truncate to ~300 chars
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."

            # Skip empty or irrelevant rows
            if not title or not url:
                continue
            if "duckduckgo.com" in url:
                continue

            results.append(WebSearchResult(title=title, url=url, snippet=snippet))

        return results


# ---------------------------------------------------------------------------
# WebSearcher facade
# ---------------------------------------------------------------------------


class WebSearcher:
    """Facade for web search with automatic backend selection.

    Currently only DuckDuckGo Lite is supported (zero dependencies).
    Additional backends (Tavily, SerpAPI, Bing) can be added by
    implementing ``BaseWebSearchBackend``.

    Usage::

        searcher = WebSearcher(max_results=5, timeout=10.0)
        response = searcher.search("招商银行 个人贷款")
        for r in response.results:
            print(f"{r.title}: {r.snippet[:80]}...")
    """

    def __init__(
        self,
        backend: str = "auto",
        max_results: int = 5,
        timeout: float = 10.0,
    ):
        self.max_results = max_results
        self.timeout = timeout

        if backend == "auto":
            self._backend = DuckDuckGoBackend(timeout=timeout)
        elif backend == "duckduckgo":
            self._backend = DuckDuckGoBackend(timeout=timeout)
        else:
            raise ValueError(f"Unknown web search backend: {backend}")

        logger.info("WebSearcher ready (backend=%s, max_results=%d)",
                     type(self._backend).__name__, max_results)

    def search(self, query: str) -> WebSearchResponse:
        """Execute a web search.

        Returns:
            WebSearchResponse with results (may be empty on failure).
            Never raises — errors are logged and an empty response returned.
        """
        try:
            return self._backend.search(
                query, max_results=self.max_results, region="cn-zh"
            )
        except Exception:
            logger.exception("Web search failed for query: %s", query[:80])
            return WebSearchResponse(
                backend=type(self._backend).__name__,
                query=query,
            )
