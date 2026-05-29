# Research RAG — Slide Deck

A complete, presentable slide deck explaining the Research RAG system.

## Files

| File | What it is |
|---|---|
| `research-rag-deck.html` | **Self-contained reveal.js deck** with live-rendered Mermaid diagrams. Just open it in a browser. |
| `research-rag-deck.md` | Same content as **Marp Markdown** — easy to edit, version, or render with Marp/Slidev. |

## How to present (HTML)

Open `research-rag-deck.html` in any modern browser.

- **Arrow keys** — navigate
- **F** — fullscreen
- **S** — speaker notes view
- **Esc / O** — slide overview
- **? ** — keyboard help

> The HTML loads reveal.js + Mermaid from a CDN, so the first open needs an internet connection.

### Export to PDF
Append `?print-pdf` to the URL and use the browser's Print → Save as PDF:

```
research-rag-deck.html?print-pdf
```

## How to render the Markdown (Marp)

```bash
# install once
npm i -g @marp-team/marp-cli

# to PDF / PPTX / HTML
marp slides/research-rag-deck.md --pdf
marp slides/research-rag-deck.md --pptx
marp slides/research-rag-deck.md --html
```

> Mermaid in Marp needs a Mermaid plugin/theme; the HTML deck renders Mermaid out of the box.
