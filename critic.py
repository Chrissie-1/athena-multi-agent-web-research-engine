"""Critic agent: the self-reflection step that makes the second draft worth reading."""

from utils import call_llm

CRITIC_PROMPT = """You are a ruthless academic reviewer. Review this report for:
1. Unsupported claims - any fact stated without a [Source X] citation, or a citation that
   looks like it was invented.
2. Missing angles - efficiency, scalability, cost, limitations, ethics, or counter-evidence
   the report ignores.
3. Vague statements - hedging like "significantly improves" with no number or mechanism.

Give exactly 3 bullet points of feedback. Each bullet must quote or name the specific passage
it is about and say concretely what the writer should change. Do not rewrite the report yourself.

Query: {query}

Draft:
{draft}

Critique:"""


def critique_draft(query: str, draft: str) -> str:
    return call_llm(CRITIC_PROMPT.format(query=query, draft=draft), temperature=0.4)
