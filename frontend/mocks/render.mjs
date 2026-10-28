// Renders every screen in mocks.html to png/. Run from the repo root: node frontend/mocks/render.mjs
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

// require, not import: it resolves a globally installed Playwright, which ESM import doesn't.
const { chromium } = createRequire(import.meta.url)("playwright");

const dir = path.dirname(fileURLToPath(import.meta.url));
const url = `file://${path.join(dir, "mocks.html")}`;

const browser = await chromium.launch();
const desktop = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1.5 });
await desktop.goto(url);
const screens = await desktop.evaluate(() => window.SCREENS);

async function shoot(page, screen, mobile, file) {
  await page.goto(`${url}?screen=${screen}${mobile ? "&mobile=1" : ""}`);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: path.join(dir, "png", file), fullPage: !screen.includes("discard") });
}

for (const s of screens) await shoot(desktop, s, false, `${s}.png`);

const phone = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
for (const s of ["02-enroll-typing", "07-identify-match", "10-auth-verified"]) await shoot(phone, s, true, `m-${s}.png`);

// Style study: the identify-match screen in Frutiger Aero, for comparison.
await desktop.goto(`file://${path.join(dir, "aero.html")}`);
await desktop.evaluate(() => document.fonts.ready);
await desktop.screenshot({ path: path.join(dir, "png", "aero-07-identify-match.png"), fullPage: true });

await browser.close();
console.log(`rendered ${screens.length} desktop + 3 mobile screens + 1 aero study`);
