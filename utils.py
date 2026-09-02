"""LLM client, HTML cleaning and chunking helpers shared by every agent."""

import json
import os
import re
import time
from typing import Iterator, List, Optional

from dotenv import load_dotenv
from groq import Groq

# secrets.env is the local convention; .env works too, and on HF Spaces the key
# arrives as a repository secret already in the environment.
load_dotenv("secrets.env")
load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy secrets.env.example to secrets.env "
                "and add a free key from https://console.groq.com"
            )
        _client = Groq(api_key=key)
    return _client


def call_llm(
    prompt: str,
    json_mode: bool = False,
    stream: bool = False,
    temperature: float = 0.3,
    max_retries: int = 3,
):
    """Call Groq. Returns a string, or a token iterator when stream=True.

    The free tier allows 30 requests/minute, so transient failures are retried
    with a short backoff rather than crashing a five-minute research run.
    """
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            if stream:
                return _stream_tokens(get_client().chat.completions.create(stream=True, **kwargs))
            response = get_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # rate limits, 5xx, dropped connections
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 2
            print(f"[llm] {type(exc).__name__}: {exc} - retrying in {wait}s")
            time.sleep(wait)
    return ""


def _stream_tokens(completion) -> Iterator[str]:
    for part in completion:
        token = part.choices[0].delta.content
        if token:
            yield token


def extract_json(text: str) -> dict:
    """Parse JSON out of an LLM reply, even when it is fenced or prefixed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def clean_text(html: str) -> str:
    """Extract readable article text; trafilatura first, BeautifulSoup as fallback."""
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted and len(extracted) > 200:
            return extracted
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception:
        return ""


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Split on paragraphs, then pack into ~`size`-char windows with overlap.

    Paragraph-first keeps sentences intact; the overlap stops a fact from being
    cut in half across two chunks and losing its context.
    """
    text = re.sub(r"\s+\n", "\n", (text or "").strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = max(size - overlap, 1)
            for i in range(0, len(para), step):
                piece = para[i : i + size]
                if len(piece) > 80:
                    chunks.append(piece)
        elif len(buffer) + len(para) + 2 <= size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if len(c) > 80]
