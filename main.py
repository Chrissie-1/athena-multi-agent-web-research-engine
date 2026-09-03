"""Orchestrator: Planner -> Searcher -> Reader -> Writer -> Critic -> Reflection."""

import re
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple

from critic import critique_draft
from models import ResearchState, SubQuery
from planner import plan_research
from reader import fetch_and_chunk
from retriever import retrieve_relevant_chunks
from searcher import search_web
from utils import normalize_citations
from writer import build_source_registry, format_references, revise_report, write_draft

# Status lines carry emoji; a default Windows console is cp1252 and would raise.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAX_RESULTS_PER_QUERY = 5
CHUNKS_PER_SUB_QUERY = 3


def research_stream(query: str) -> Iterator[Tuple[str, ResearchState]]:
    """Run the full pipeline, yielding (status line, state) after every step."""
    state = ResearchState(user_query=query)
    started = time.time()

    yield "🚀 **Planning** — splitting your question into research angles...", state
    state.sub_queries = [SubQuery(question=q) for q in plan_research(query)]
    total = len(state.sub_queries)
    print(f"[plan] {total} sub-questions")

    for i, sq in enumerate(state.sub_queries, start=1):
        yield f"🔍 **Searching** ({i}/{total}) — {sq.question}", state
        sq.urls = search_web(sq.question, max_results=MAX_RESULTS_PER_QUERY)
        print(f"[search] {len(sq.urls)} urls for: {sq.question}")

        yield f"📖 **Reading** ({i}/{total}) — {len(sq.urls)} sources...", state
        collected = []
        for url in sq.urls:
            collected.extend(fetch_and_chunk(url))

        top = retrieve_relevant_chunks(collected, sq.question, top_k=CHUNKS_PER_SUB_QUERY)
        sq.chunks = [chunk for chunk, _ in top]
        sq.sources = [url for _, url in top]
        print(f"[retrieve] kept {len(top)}/{len(collected)} chunks")

    if not any(sq.chunks for sq in state.sub_queries):
        state.final_report = (
            "## No usable sources\n\n"
            "Every search or fetch for this question came back empty — usually a DuckDuckGo "
            "rate limit or sites blocking the scraper. Wait a minute and try again, or "
            "rephrase the question."
        )
        yield "⚠️ **No sources retrieved**", state
        return

    context, state.references = build_source_registry(state.sub_queries)

    yield "✍️ **Drafting** — writing the first report with citations...", state
    state.draft_report = normalize_citations(write_draft(query, state.sub_queries))

    yield "🧐 **Critiquing** — checking the draft against the evidence...", state
    state.critique = critique_draft(query, state.draft_report, context)

    yield "🪞 **Reflecting** — rewriting the report to answer the critique...", state
    for token in revise_report(query, state.draft_report, state.critique, context, stream=True):
        state.final_report += token
        yield "🪞 **Reflecting** — rewriting the report to answer the critique...", state

    state.final_report = normalize_citations(state.final_report.strip())
    state.final_report += format_references(state.references)

    elapsed = time.time() - started
    yield (
        f"✅ **Done** in {elapsed:.0f}s — {total} sub-questions, "
        f"{len(state.references)} sources, 1 self-revision."
    ), state


def research(query: str) -> ResearchState:
    """Blocking version for scripts and tests."""
    state = ResearchState(user_query=query)
    for status, state in research_stream(query):
        print(status.replace("**", ""))
    return state


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "report"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Research question: ")
    result = research(question)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{_slug(question)}.md"
    path.write_text(f"# {question}\n\n{result.final_report}\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(result.final_report)
    print("=" * 70)
    print(f"saved -> {path}")
