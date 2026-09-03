"""Run the pipeline and dump every intermediate stage to one file.

    python capture_run.py "your question"

Writes artifacts.md: the plan, the URLs searched, the exact chunk text that was
retrieved, the global reference numbering, the draft, the critique, and the final
report. The chunk text is the point - without it you cannot tell a grounded claim
from a fluent one.
"""

import io
import sys

from main import research
from writer import build_source_registry

DEFAULT = "What are the latest breakthroughs in Mixture of Experts?"


def dump(state) -> str:
    out = io.StringIO()
    w = out.write

    w("=== SUB-QUESTIONS ===\n")
    for i, sq in enumerate(state.sub_queries, 1):
        w(f"{i}. {sq.question}\n")
        for url in sq.urls:
            w(f"     searched: {url}\n")
    w("\n=== EVIDENCE (the only text the writer saw) ===\n")
    for i, sq in enumerate(state.sub_queries, 1):
        for chunk, url in zip(sq.chunks, sq.sources):
            w(f"\n--- sub-question {i} <- {url}\n{chunk}\n")

    w("\n=== REFERENCES (global numbering) ===\n")
    for i, url in enumerate(state.references, 1):
        w(f"[Source {i}] {url}\n")

    w("\n=== DRAFT ===\n" + state.draft_report)
    w("\n\n=== CRITIQUE ===\n" + state.critique)
    w("\n\n=== FINAL ===\n" + state.final_report + "\n")
    return out.getvalue()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or DEFAULT
    state = research(query)
    if not state.sub_queries:
        sys.exit("pipeline produced no sub-questions")

    text = dump(state)
    io.open("artifacts.md", "w", encoding="utf-8").write(text)

    context, refs = build_source_registry(state.sub_queries)
    print(f"\nwrote artifacts.md ({len(text)} chars)")
    print(f"evidence: {len(context)} chars across {len(refs)} sources")
    print(f"draft: {len(state.draft_report)} chars | final: {len(state.final_report)} chars")
