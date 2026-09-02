"""Writer agent: turn retrieved evidence into a cited report, then revise it."""

from typing import List, Tuple

from models import SubQuery
from utils import call_llm


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


DRAFT_PROMPT = """You are a technical research writer. Write a comprehensive, structured report
answering: "{query}"

Rules:
- Use ONLY the provided context. If the context does not cover something, say so explicitly
  instead of inventing it.
- Cite every factual claim inline as [Source X], where X is the source number given in the context.
- Do not invent source numbers, and do not add a references list (it is appended automatically).
- Be concrete: prefer numbers, model names, and dates over adjectives.

Structure the report with exactly these headings:
## Summary
## Key Findings
## Technical Details
## Future Implications

Context:
{context}

Report:"""


REVISE_PROMPT = """You are the same technical research writer, revising after peer review.

Original question: "{query}"

Your draft:
{draft}

Reviewer's critique:
{critique}

Rewrite the report so that every critique point is addressed. Keep the same four headings
(## Summary, ## Key Findings, ## Technical Details, ## Future Implications) and the same
[Source X] citation format with the same source numbers. Where the reviewer flags a claim the
evidence does not support, either remove it or mark it explicitly as a limitation rather than
inventing a new citation. Do not mention the review process itself.

Revised report:"""


def write_draft(query: str, sub_queries: List[SubQuery], stream: bool = False):
    context, _ = build_source_registry(sub_queries)
    return call_llm(DRAFT_PROMPT.format(query=query, context=context), stream=stream)


def revise_report(query: str, draft: str, critique: str, stream: bool = False):
    return call_llm(
        REVISE_PROMPT.format(query=query, draft=draft, critique=critique), stream=stream
    )


def format_references(references: List[str]) -> str:
    if not references:
        return ""
    lines = "\n".join(f"{i}. <{url}>" for i, url in enumerate(references, start=1))
    return f"\n\n## References\n{lines}\n"
