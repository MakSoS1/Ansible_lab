import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const outDir = path.resolve('artifacts/raw');
fs.mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  recordVideo: { dir: outDir, size: { width: 1920, height: 1080 } },
});
const page = await context.newPage();
await page.goto('http://127.0.0.1:5173/#/showcase', { waitUntil: 'networkidle' });
await page.waitForTimeout(30000);
const video = page.video();
await page.close();
await context.close();
const recorded = await video.path();
fs.mkdirSync('artifacts', { recursive: true });
fs.copyFileSync(recorded, 'artifacts/showcase.webm');
await browser.close();
console.log('recorded', recorded);
