"""Planner agent: turns one broad question into three searchable sub-questions."""

from typing import List

from utils import call_llm, extract_json

PLANNER_PROMPT = """You are a research planner. Break the user's question into exactly 3 specific,
search-engine-optimized sub-questions that cover different angles (for example: technical mechanism,
performance and benchmarks, real-world applications or limitations).

Rules:
- Each sub-question must stand alone as a web search query (no pronouns, no "it", no "this").
- Do not repeat the user's wording verbatim across all three.
- Output ONLY valid JSON in this exact format:
{{"sub_questions": ["Q1", "Q2", "Q3"]}}

User Query: {query}"""


def plan_research(query: str) -> List[str]:
    raw = call_llm(PLANNER_PROMPT.format(query=query), json_mode=True)
    data = extract_json(raw)
    questions = [q.strip() for q in data.get("sub_questions", []) if isinstance(q, str) and q.strip()]

    if not questions:
        # Never let a malformed plan kill the run - fall back to the raw query.
        return [query]
    return questions[:3]


if __name__ == "__main__":
    for q in [
        "What are the latest breakthroughs in Mixture of Experts?",
        "why do cats knock things off tables",
        "Is Rust actually faster than Go for network services in 2026?",
    ]:
        print(f"\n=== {q}")
        for sub in plan_research(q):
            print(f"  - {sub}")
