"""Searcher agent: sub-question -> list of candidate URLs (DuckDuckGo, no API key)."""

from typing import List
from urllib.parse import urlparse

# The package was renamed duckduckgo-search -> ddgs; support both.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - older installs
    from duckduckgo_search import DDGS

# DuckDuckGo rate-limits aggressively. If a live demo hits that wall we still want
# a real report, so the two showcase queries have known-good URLs to fall back on.
FALLBACK_URLS = {
    "mixture of experts": [
        "https://huggingface.co/blog/moe",
        "https://arxiv.org/abs/2401.04088",
        "https://en.wikipedia.org/wiki/Mixture_of_experts",
        "https://arxiv.org/abs/2101.03961",
    ],
    "sparse attention": [
        "https://arxiv.org/abs/2004.05150",
        "https://arxiv.org/abs/2205.14135",
        "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
        "https://arxiv.org/abs/1904.10509",
    ],
}

# Aggregators and paywalls that reliably return 403 or JS-only shells.
BLOCKED_HOSTS = (
    "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "tiktok.com", "linkedin.com", "reddit.com",
)


def _usable(url: str) -> bool:
    if not url.startswith("http"):
        return False
    host = (urlparse(url).netloc or "").lower()
    return not any(blocked in host for blocked in BLOCKED_HOSTS)


def _fallback(query: str) -> List[str]:
    lowered = query.lower()
    for topic, urls in FALLBACK_URLS.items():
        if all(word in lowered for word in topic.split()):
            print(f"[search] using offline fallback URLs for '{topic}'")
            return urls
    return []


def search_web(query: str, max_results: int = 5) -> List[str]:
    """Return up to `max_results` scrapeable URLs for a sub-question."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results * 2))
    except Exception as exc:
        print(f"[search] DuckDuckGo failed ({type(exc).__name__}: {exc})")
        return _fallback(query)[:max_results]

    urls, seen = [], set()
    for result in results:
        url = result.get("href") or result.get("url") or ""
        host = urlparse(url).netloc
        if _usable(url) and host not in seen:
            seen.add(host)  # one page per domain keeps the evidence diverse
            urls.append(url)
        if len(urls) == max_results:
            break

    return urls or _fallback(query)[:max_results]


if __name__ == "__main__":
    for url in search_web("Mixture of Experts routing breakthroughs 2025"):
        print(url)
