"""Reader agent: fetch a URL, strip it to article text, chunk it, keep the source."""

from typing import List, Tuple

import requests

from utils import chunk_text, clean_text

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_BYTES = 2_000_000  # skip giant pages instead of stalling the run


def fetch_and_chunk(url: str, timeout: int = 10) -> List[Tuple[str, str]]:
    """Return [(chunk, url), ...]. One dead link must never tank the report."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            print(f"[read] skip non-html {url} ({content_type})")
            return []
        if len(response.content) > MAX_BYTES:
            print(f"[read] skip oversized {url}")
            return []

        text = clean_text(response.text)
        chunks = chunk_text(text)
        print(f"[read] {len(chunks):>3} chunks from {url}")
        return [(chunk, url) for chunk in chunks]
    except Exception as exc:
        print(f"[read] failed {url} ({type(exc).__name__}: {exc})")
        return []
