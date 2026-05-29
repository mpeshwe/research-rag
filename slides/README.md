# Research RAG — Slide Deck

A complete, presentable slide deck explaining the Research RAG system —
14 slides with live Mermaid diagrams and per-slide speaker notes.

## Files

| File | What it is |
|---|---|
| `research-rag-deck.html` | **Self-contained reveal.js deck** with live-rendered Mermaid diagrams + speaker notes. Just open it in a browser. |
| `research-rag-deck.md` | Same content as **Marp Markdown** — easy to edit, version, or render with Marp/Slidev. |
| `export/research-rag-deck.pdf` | **Ready-to-upload PDF** (14 pages, 16:9) rendered from the HTML deck. |
| `export/slide-NN.png` | **Ready-to-upload PNGs** — one per slide (1280×720 @2x), great for a LinkedIn carousel. |
| `render.mjs` | Script that drives headless Chrome to screenshot each slide and assemble the PDF. |

## Speaker notes

Every slide has narration notes embedded as `<aside class="notes">`. Press **S**
in the HTML deck to open the speaker view (current slide, next slide, notes, timer).
The notes do **not** appear on the slides themselves or in the exported PDF/PNGs.

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

## Re-rendering the PDF / PNGs

The `export/` folder is pre-generated, but you can rebuild it after editing the deck:

```bash
cd slides
npm install                 # one-time: installs puppeteer-core + pdf-lib
node render.mjs             # writes export/slide-*.png and export/research-rag-deck.pdf
```

`render.mjs` loads the HTML deck in headless Chrome, waits for Mermaid to finish
rendering, screenshots each slide, and stitches the PNGs into a single PDF.

> It uses the Chrome that ships with Puppeteer. If it isn't present, run
> `npx puppeteer browsers install chrome` first.
