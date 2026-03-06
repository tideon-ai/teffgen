"""
URL Fetch tool for retrieving webpage content.

Fetches and extracts text from web pages using requests + BeautifulSoup
(both free/open source). Falls back to stdlib urllib if packages unavailable.
"""

import logging
import re
from html.parser import HTMLParser
from typing import Any


def _get_user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__} (URL Fetch Tool)"

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class _SimpleHTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter using stdlib."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._text_parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self._text_parts)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class URLFetchTool(BaseTool):
    """
    URL content fetching tool.

    Fetches webpage content and extracts readable text.

    Features:
    - Fetch and extract text from web pages
    - Configurable max content length
    - Domain allowlist/blocklist
    - Timeout control
    - Uses requests + beautifulsoup4 if available, falls back to urllib

    Note:
        This tool makes HTTP requests to external websites.
        Configure allowed_domains to restrict access.
    """

    def __init__(
        self,
        allowed_domains: set[str] | None = None,
        blocked_domains: set[str] | None = None,
        max_content_length: int = 10000,
        timeout: int = 15,
    ):
        """
        Initialize the URL Fetch tool.

        Args:
            allowed_domains: If set, only fetch from these domains.
            blocked_domains: Domains to never fetch from.
            max_content_length: Max characters to return (default: 10000).
            timeout: Request timeout in seconds (default: 15).
        """
        super().__init__(
            metadata=ToolMetadata(
                name="url_fetch",
                description=(
                    "Fetch a webpage and extract its text content. "
                    "Returns the readable text from a URL. "
                    "Use this to get information from specific web pages."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="The URL to fetch (must start with http:// or https://)",
                        required=True,
                        min_length=8,
                    ),
                    ParameterSpec(
                        name="extract_links",
                        type=ParameterType.BOOLEAN,
                        description="Also extract links from the page",
                        required=False,
                        default=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                timeout_seconds=timeout,
                tags=["url", "web", "fetch", "http", "scraping"],
                examples=[
                    {
                        "url": "https://example.com",
                        "output": {"title": "Example Domain", "text": "This domain is for use in..."},
                    },
                ],
            )
        )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains or set()
        self.max_content_length = max_content_length
        self.timeout = timeout

        logger.info(
            "\u2139\ufe0f  URL Fetch tool makes HTTP requests to external websites.\n"
            "    Configure allowed_domains to restrict access."
        )

    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Extract domain
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        if self.allowed_domains and domain not in self.allowed_domains:
            raise ValueError(f"Domain '{domain}' not in allowed domains list")
        if domain in self.blocked_domains:
            raise ValueError(f"Domain '{domain}' is blocked")

        return url

    def _extract_with_beautifulsoup(self, html: str) -> tuple:
        """Extract text using BeautifulSoup."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        links = []
        for a in soup.find_all("a", href=True):
            links.append({"text": a.get_text(strip=True), "href": a["href"]})

        return title, text, links

    def _extract_with_stdlib(self, html: str) -> tuple:
        """Extract text using stdlib HTML parser."""
        extractor = _SimpleHTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        # Try to find title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Extract links
        links = []
        for m in re.finditer(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            links.append({"text": re.sub(r'<[^>]+>', '', m.group(2)).strip(), "href": m.group(1)})

        return title, text, links

    async def _execute(
        self,
        url: str,
        extract_links: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch URL and extract text."""
        url = self._validate_url(url)

        # Try requests first, fall back to urllib
        html = ""
        try:
            import requests
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": _get_user_agent()},
            )
            resp.raise_for_status()
            html = resp.text
        except ImportError:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": _get_user_agent()})
            with urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")

        # Extract text
        try:
            title, text, links = self._extract_with_beautifulsoup(html)
        except ImportError:
            title, text, links = self._extract_with_stdlib(html)

        # Truncate
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n... (content truncated)"

        result = {
            "url": url,
            "title": title,
            "text": text,
            "content_length": len(text),
        }

        if extract_links:
            result["links"] = links[:50]  # Limit links

        return result
