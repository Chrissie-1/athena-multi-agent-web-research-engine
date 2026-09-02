"""Athena - Gradio dashboard for the self-reflective research agent."""

import inspect
import os
import traceback

import gradio as gr

from main import research_stream
from models import ResearchState

CSS = """
#report { min-height: 320px; padding: 4px 8px; }
#report h2 { margin-top: 1.2em; border-bottom: 1px solid var(--border-color-primary); padding-bottom: 4px; }
.stage-log textarea { font-family: var(--font-mono); }
footer { visibility: hidden; }
"""

INTRO = """# 🧠 Athena — Self-Reflective Research Agent
Ask a complex question. Athena plans the angles, searches the live web, reads and ranks the
evidence, drafts a cited report — then **critiques its own draft and rewrites it**.

*Planner → Searcher → Reader (RAG) → Writer → Critic → Reflection*
"""

EXAMPLES = [
    "What are the latest breakthroughs in Mixture of Experts?",
    "What are the latest breakthroughs in sparse attention?",
    "How do vector databases handle billion-scale similarity search?",
    "Is Rust displacing Go for high-throughput network services?",
]


def _trace_md(state: ResearchState) -> str:
    if not state.sub_queries:
        return "_Planning..._"
    parts = ["### Research plan"]
    for i, sq in enumerate(state.sub_queries, start=1):
        parts.append(f"**{i}. {sq.question}**")
        if sq.urls:
            parts += [f"- {url}" for url in sq.urls]
        else:
            parts.append("- _searching..._")
        if sq.sources:
            kept = sorted(set(sq.sources))
            parts.append(f"- ✅ kept {len(sq.chunks)} chunks from {len(kept)} source(s)")
        parts.append("")
    return "\n".join(parts)


def _review_md(state: ResearchState) -> str:
    if not state.critique:
        return "_The Critic runs after the first draft._"
    return (
        "### 🧐 Critic's feedback on the first draft\n"
        f"{state.critique}\n\n---\n\n### 📝 First draft (pre-revision)\n"
        f"{state.draft_report}"
    )


def run_research(query: str):
    query = (query or "").strip()
    if not query:
        yield "⚠️ Enter a question first.", "", "", ""
        return

    report_so_far = ""
    try:
        for status, state in research_stream(query):
            # The Reflection step yields per token; only repaint on real progress.
            grown = len(state.final_report) - len(report_so_far)
            if grown and grown < 40 and not status.startswith("✅"):
                continue
            report_so_far = state.final_report
            yield status, state.final_report, _trace_md(state), _review_md(state)
    except Exception as exc:
        traceback.print_exc()
        yield (
            f"❌ **{type(exc).__name__}** — {exc}",
            report_so_far,
            "",
            "",
        )


# Gradio 6 moved theme/css from the Blocks constructor to launch(); support both.
_LAUNCH_TAKES_THEME = "theme" in inspect.signature(gr.Blocks.launch).parameters
_STYLE = {"theme": gr.themes.Soft(), "css": CSS}
_BLOCKS_KWARGS = {"title": "Athena Research Agent"}
if not _LAUNCH_TAKES_THEME:
    _BLOCKS_KWARGS.update(_STYLE)

with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        query_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What are the latest breakthroughs in sparse attention?",
            scale=5,
            autofocus=True,
        )
        submit_btn = gr.Button("Generate Report", variant="primary", scale=1)

    status = gr.Markdown("_Idle — ask something to start._", elem_classes="stage-log")

    with gr.Tabs():
        with gr.Tab("📄 Report"):
            output_md = gr.Markdown(elem_id="report")
        with gr.Tab("🔎 Research trace"):
            trace_md = gr.Markdown("_Sub-questions and sources appear here._")
        with gr.Tab("🧐 Critique & first draft"):
            review_md = gr.Markdown("_The self-review appears here._")

    gr.Examples(examples=EXAMPLES, inputs=query_box, label="Try one")
    gr.Markdown(
        "Runs on Groq (`llama-3.3-70b-versatile`), DuckDuckGo and local "
        "`all-MiniLM-L6-v2` embeddings. A full run takes 60–120s."
    )

    outputs = [status, output_md, trace_md, review_md]
    submit_btn.click(run_research, inputs=query_box, outputs=outputs)
    query_box.submit(run_research, inputs=query_box, outputs=outputs)

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"}
    if _LAUNCH_TAKES_THEME:
        launch_kwargs.update(_STYLE)
    demo.queue().launch(**launch_kwargs)
