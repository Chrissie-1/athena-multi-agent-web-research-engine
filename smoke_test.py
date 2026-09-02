"""End-to-end pipeline check with the LLM stubbed out.

Exercises the real Searcher, Reader, chunker and FAISS retriever (live network),
and a fake Groq client, so the orchestration can be verified without an API key
or burning free-tier requests. Run: python smoke_test.py
"""

import json
import sys

import critic
import planner
import writer
from utils import chunk_text, extract_json

QUERY = "What are the latest breakthroughs in Mixture of Experts?"

calls = []


def fake_llm(prompt, json_mode=False, stream=False, temperature=0.3, max_retries=3):
    calls.append(prompt)
    if json_mode:
        return "```json\n" + json.dumps({"sub_questions": [
            "Mixture of Experts sparse routing architecture explained",
            "Mixture of Experts model benchmark performance 2025",
            "Mixture of Experts inference cost and deployment limitations",
        ]}) + "\n```"  # fenced on purpose: extract_json must survive it
    if "Critique:" in prompt:
        return "- Claim in ## Summary has no citation.\n- No cost analysis.\n- 'Much faster' is vague."
    body = "## Summary\nRouting is sparse [Source 1].\n## Key Findings\n- Fewer active params [Source 2].\n## Technical Details\nTop-k gating [Source 1].\n## Future Implications\nCheaper serving [Source 3]."
    return iter(body.split(" ")) if stream else body


def main() -> int:
    # Unit checks that need no network.
    assert extract_json('noise {"a": 1} trailing') == {"a": 1}
    chunks = chunk_text("para one. " * 60 + "\n\n" + "para two. " * 60)
    assert chunks and all(len(c) <= 500 for c in chunks), [len(c) for c in chunks]
    print(f"[unit] extract_json + chunk_text ok ({len(chunks)} chunks)")

    planner.call_llm = fake_llm
    writer.call_llm = fake_llm
    critic.call_llm = fake_llm

    from main import research_stream  # imported late so the patches are in place

    state = None
    for status, state in research_stream(QUERY):
        if not status.startswith("🪞"):
            print("  " + status.replace("**", ""))

    assert len(state.sub_queries) == 3, state.sub_queries
    scraped = sum(len(sq.chunks) for sq in state.sub_queries)
    print(f"[pipeline] {scraped} chunks retrieved, {len(state.references)} sources cited")

    if scraped == 0:
        print("[warn] no pages could be scraped - network or rate limit. Pipeline itself ran.")
        return 0

    assert state.draft_report and state.critique, "writer/critic did not run"
    assert "## References" in state.final_report, state.final_report[-200:]
    assert state.references[0].startswith("http"), state.references
    context, refs = writer.build_source_registry(state.sub_queries)
    assert len(refs) == len(set(refs)), "reference numbering is not deduplicated"
    assert f"[Source {len(refs)}:" in context, "source numbering did not reach the last ref"
    print(f"[pipeline] citations numbered 1..{len(refs)} globally, references section present")
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
