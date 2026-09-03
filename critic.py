"""Critic agent: the self-reflection step that makes the second draft worth reading.

The critic reads the retrieved evidence alongside the draft. Without it, it cannot tell a
supported claim from an unsupported one, and its suggestions become new hallucinations.
"""

from utils import call_llm, load_prompt, render


def critique_draft(query: str, draft: str, context: str) -> str:
    prompt = render(load_prompt("critic"), query=query, context=context, draft=draft)
    return call_llm(prompt, temperature=0.2)
