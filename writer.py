"""Writer agent: turn retrieved evidence into a cited report, then revise it."""

from typing import List, Tuple

from models import SubQuery
from utils import call_llm, load_prompt, render


def build_source_registry(sub_queries: List[SubQuery]) -> Tuple[str, List[str]]:
    """Number every URL once across the whole report, then lay out the evidence.

    A per-sub-question numbering would make [Source 1] mean three different pages,
    so citations are global: reference i is always the same URL everywhere.
    """
    references: List[str] = []
    index_of = {}
    for sq in sub_queries:
        for url in sq.sources:
            if url not in index_of:
                references.append(url)
                index_of[url] = len(references)

    blocks = []
    for sq in sub_queries:
        blocks.append(f"\n### Sub-Question: {sq.question}")
        for chunk, url in zip(sq.chunks, sq.sources):
            blocks.append(f"[Source {index_of[url]}: {url}]\n{chunk}\n")
    return "\n".join(blocks), references


def write_draft(query: str, sub_queries: List[SubQuery], stream: bool = False):
    context, _ = build_source_registry(sub_queries)
    prompt = render(load_prompt("writer_draft"), query=query, context=context)
    return call_llm(prompt, stream=stream)


def revise_report(query: str, draft: str, critique: str, context: str, stream: bool = False):
    """The reviser gets the evidence too, so it can refuse a finding the evidence contradicts."""
    prompt = render(
        load_prompt("writer_revise"), query=query, context=context, draft=draft, critique=critique
    )
    return call_llm(prompt, stream=stream)


def format_references(references: List[str]) -> str:
    if not references:
        return ""
    lines = "\n".join(f"{i}. <{url}>" for i, url in enumerate(references, start=1))
    return f"\n\n## References\n{lines}\n"
