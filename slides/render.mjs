import puppeteer from 'puppeteer-core';
import { PDFDocument } from 'pdf-lib';
import fs from 'node:fs/promises';
import path from 'node:path';

const DECK = 'file://' + path.resolve('research-rag-deck.html');
const OUT = path.resolve('export');
const W = 1280, H = 720, SCALE = 2;

// Locate the Chrome that `puppeteer browsers install chrome` placed in the cache.
async function findChrome() {
  const base = path.join(process.env.HOME, '.cache/puppeteer/chrome');
  const dirs = await fs.readdir(base);
  for (const d of dirs) {
    const p = path.join(base, d, 'chrome-linux64', 'chrome');
    try { await fs.access(p); return p; } catch {}
  }
  throw new Error('Chrome not found in puppeteer cache');
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: await findChrome(),
  headless: 'new',
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--force-color-profile=srgb', '--ignore-certificate-errors'],
});

const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: SCALE });

await page.goto(DECK, { waitUntil: 'networkidle0', timeout: 120000 });

// Wait for Reveal to exist and every Mermaid block to be replaced with an <svg>.
await page.waitForFunction(() => {
  const blocks = [...document.querySelectorAll('.mermaid')];
  const rendered = blocks.every(b => b.querySelector('svg'));
  return window.Reveal && window.Reveal.isReady && window.Reveal.isReady() && rendered;
}, { timeout: 120000 });

await sleep(800);

const total = await page.evaluate(() => window.Reveal.getTotalSlides());
console.log(`Rendering ${total} slides…`);

await fs.mkdir(OUT, { recursive: true });

const pngPaths = [];
for (let i = 0; i < total; i++) {
  await page.evaluate((idx) => window.Reveal.slide(idx, 0, 0), i);
  await sleep(450); // let transitions/layout settle
  const file = path.join(OUT, `slide-${String(i + 1).padStart(2, '0')}.png`);
  await page.screenshot({ path: file });
  pngPaths.push(file);
  console.log(`  ✓ ${path.basename(file)}`);
}

await browser.close();

// Assemble the PNGs into a single PDF, one slide per page.
const pdf = await PDFDocument.create();
for (const p of pngPaths) {
  const bytes = await fs.readFile(p);
  const img = await pdf.embedPng(bytes);
  const pageW = W, pageH = H; // logical points (16:9)
  const pg = pdf.addPage([pageW, pageH]);
  pg.drawImage(img, { x: 0, y: 0, width: pageW, height: pageH });
}
const pdfBytes = await pdf.save();
const pdfPath = path.join(OUT, 'research-rag-deck.pdf');
await fs.writeFile(pdfPath, pdfBytes);
console.log(`\nPDF → ${pdfPath}`);
console.log(`PNGs → ${OUT}/slide-*.png`);
