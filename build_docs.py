"""Wrap the project page source into a standalone document for GitHub Pages.

The Artifact runtime supplies the <!doctype>/<head> skeleton and a small reset;
GitHub Pages does not. Keeping one source and generating the other stops the two
copies of the page from drifting apart.

    python build_docs.py path/to/athena.html
"""

import io
import sys

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Athena: a six-agent research engine that plans, searches, reads and cites the live web, then critiques and rewrites its own report.">
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
"""

REPO = "https://github.com/Chrissie-1/athena-research-agent"

src = io.open(sys.argv[1], encoding="utf-8").read()
if REPO not in src:
    src = src.replace(
        "<span>MIT</span>",
        f'<span>MIT</span>\n    <span><a href="{REPO}">{REPO.removeprefix("https://")}</a></span>',
        1,
    )

body_at = src.index('<div class="page">')
io.open("docs/index.html", "w", encoding="utf-8", newline="\n").write(
    HEAD + src[:body_at] + "</head>\n<body>\n" + src[body_at:] + "\n</body>\n</html>\n"
)
print(f"docs/index.html regenerated from {sys.argv[1]}")
