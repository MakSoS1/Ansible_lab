import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const outDir = path.resolve('artifacts/raw');
fs.mkdirSync(outDir, { recursive: true });

function intersects(a, b, pad = 0) {
  return !(
    a.x + a.width + pad <= b.x ||
    b.x + b.width + pad <= a.x ||
    a.y + a.height + pad <= b.y ||
    b.y + b.height + pad <= a.y
  );
}

async function box(page, selector) {
  const loc = page.locator(selector).first();
  const b = await loc.boundingBox();
  if (!b) throw new Error(`Missing layout region: ${selector}`);
  return b;
}

async function assertSeparated(page, pairs, label) {
  for (const [aSel, bSel, pad = 0] of pairs) {
    const [a, b] = await Promise.all([box(page, aSel), box(page, bSel)]);
    if (intersects(a, b, pad)) {
      throw new Error(`${label}: ${aSel} overlaps ${bSel}: ${JSON.stringify({a,b})}`);
    }
  }
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  recordVideo: { dir: outDir, size: { width: 1920, height: 1080 } },
});
const page = await context.newPage();
await page.goto('http://127.0.0.1:5173/#/showcase', { waitUntil: 'networkidle' });

// Menu stage: enforce the same separation a TV/controller layout requires.
await page.waitForTimeout(10500);
await assertSeparated(page, [
  ['.ref-brand', '.ref-tab-shell', 10],
  ['.ref-tab-shell', '.ref-profile', 10],
  ['.ref-song-card.active', '.ref-selected-copy', 8],
], 'menu layout');

// Gameplay stage: every persistent control belongs on an edge.
await page.waitForTimeout(11200);
await assertSeparated(page, [
  ['.ref-song-panel', '.ref-score-panel', 12],
  ['.ref-score-panel', '.ref-next-moves', 10],
  ['.ref-next-moves', '.ref-timeline', 10],
  ['.ref-grade', '.ref-next-moves', 8],
], 'gameplay layout');

const viewport = page.viewportSize();
if (!viewport) throw new Error('No viewport');
const safe = {
  x: viewport.width * .25,
  y: viewport.height * .17,
  width: viewport.width * .50,
  height: viewport.height * .61,
};
for (const selector of ['.ref-song-panel','.ref-score-panel','.ref-next-moves','.ref-timeline']) {
  const region = await box(page, selector);
  if (intersects(region, safe, 0)) {
    throw new Error(`central dancer safe-zone violation: ${selector} ${JSON.stringify(region)} intersects ${JSON.stringify(safe)}`);
  }
}

// Finish a ~30s walkthrough after validation.
await page.waitForTimeout(8300);
const video = page.video();
await page.close();
await context.close();
const recorded = await video.path();
fs.mkdirSync('artifacts', { recursive: true });
fs.copyFileSync(recorded, 'artifacts/showcase.webm');
await browser.close();
console.log('recorded', recorded);
