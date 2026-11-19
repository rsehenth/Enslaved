"""External source discovery via real search engines (DuckDuckGo / Brave).

Only the query text leaves the machine. Failures raise `SearchUnavailable` with a clear message.
"""

from __future__ import annotations

import html as html_module
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

_UA = "Enslaved/0.2 (local assistant; personal use)"
_OFFICIAL_HOSTS = ("docs.", "developer.", "developers.", "learn.", "support.", "help.", "documentation.", "wiki.")
_OFFICIAL_PATHS = ("/docs", "/documentation", "/reference", "/manual", "/guide", "/api", "/learn", "/help")
_TRUSTED = ("github.com", "readthedocs.io", "developer.mozilla.org", "python.org", "pypi.org", "w3.org", "wikipedia.org", "arxiv.org")
_LOW_VALUE = ("reddit.com", "quora.com", "pinterest.", "facebook.com", "instagram.com", "tiktok.com", "youtube.com", "twitter.com", "x.com")


class SearchUnavailable(RuntimeError):
    """The search engine requires configuration, is rate-limited, or is offline."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    priority: int = 0

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "priority": self.priority}


def source_priority(url: str) -> int:
    """Prioritizes official documentation and de-prioritizes social media/forums."""
    parts = urlsplit(url)
    host, path = (parts.hostname or "").lower(), parts.path.lower()
    score = 0
    if host.startswith(_OFFICIAL_HOSTS):
        score += 2
    if any(path.startswith(p) or p + "/" in path for p in _OFFICIAL_PATHS):
        score += 1
    if host.endswith((".gov", ".edu", ".gov.br", ".edu.br")) or any(host == t or host.endswith("." + t) for t in _TRUSTED):
        score += 3
    if any(bad in host for bad in _LOW_VALUE):
        score -= 2
    return score


def _strip_tags(text: str) -> str:
    return html_module.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def parse_duckduckgo(page: str) -> list[SearchResult]:
    results = []
    blocks = re.findall(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="[^"]*result__a|\Z)', page, re.DOTALL)
    for href, title, rest in blocks:
        href = html_module.unescape(href)
        if "uddg=" in href:
            href = unquote(parse_qs(urlsplit(href if "//" in href else "//" + href).query).get("uddg", [""])[0])
        if not href.startswith(("http://", "https://")):
            continue
        snippet = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', rest, re.DOTALL)
        results.append(SearchResult(_strip_tags(title), href, _strip_tags(snippet.group(1)) if snippet else ""))
    return results


async def web_search(query: str, provider: str, max_results: int, timeout: float = 15.0, transport: httpx.AsyncBaseTransport | None = None) -> list[SearchResult]:
    query = query.strip()
    if not query:
        raise ValueError("Empty query.")
    async with httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport, headers={"User-Agent": _UA}) as client:
        try:
            if provider == "brave":
                key = os.environ.get("ENSLAVED_SEARCH_API_KEY", "").strip()
                if not key:
                    raise SearchUnavailable(
                        "The 'brave' provider requires ENSLAVED_SEARCH_API_KEY (set in .env) — no key was found."
                    )
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": key, "Accept": "application/json"},
                )
                if resp.status_code in (401, 403, 429):
                    raise SearchUnavailable(f"Search API rejected the request (HTTP {resp.status_code}): check key/quota.")
                resp.raise_for_status()
                items = (resp.json().get("web") or {}).get("results") or []
                found = [SearchResult(i.get("title", ""), i.get("url", ""), _strip_tags(i.get("description", ""))) for i in items]
            elif provider == "duckduckgo":
                resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
                if resp.status_code in (202, 403, 429) or "anomaly" in resp.text[:4000].lower():
                    raise SearchUnavailable("DuckDuckGo rate-limited or blocked the automated query. Try again later or use direct URLs.")
                resp.raise_for_status()
                found = parse_duckduckgo(resp.text)
            else:
                raise SearchUnavailable(f"Unknown search provider: {provider!r} (use duckduckgo or brave).")
        except httpx.HTTPError as exc:
            raise SearchUnavailable(f"Network failure querying search engine ({exc.__class__.__name__}).") from exc
    found = [r for r in found if r.url.startswith(("http://", "https://"))]
    for r in found:
        r.priority = source_priority(r.url)
    return sorted(found, key=lambda r: -r.priority)[:max_results]
