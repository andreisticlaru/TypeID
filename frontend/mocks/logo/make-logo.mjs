// Generates the neobrutalist TypeID mark (three colour variants) and a review sheet.
// Run from the repo root: node frontend/mocks/logo/make-logo.mjs, then node frontend/mocks/render.mjs.
// Same idea as the original mark (public/mark.png): two broken scan rings around a cursor, redrawn
// flat, with an ink outline, a hard offset shadow and square line ends.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));
const INK = "#161328";

// One ring as n outlined arc segments (annular sectors), so each piece gets the ink outline that a
// dashed stroke can't give its ends.
function ring(r, width, n, gapDeg, rotate, fill, outline) {
  const ro = r + width / 2, ri = r - width / 2, step = 360 / n, pt = (rad, deg) => {
    const a = ((deg + rotate - 90) * Math.PI) / 180;
    return `${(47 + rad * Math.cos(a)).toFixed(2)} ${(47 + rad * Math.sin(a)).toFixed(2)}`;
  };
  return Array.from({ length: n }, (_, i) => {
    const a0 = i * step + gapDeg / 2, a1 = (i + 1) * step - gapDeg / 2;
    return `<path d="M${pt(ro, a0)} A${ro} ${ro} 0 0 1 ${pt(ro, a1)} L${pt(ri, a1)} A${ri} ${ri} 0 0 0 ${pt(ri, a0)}Z" fill="${fill}"${outline ? ` stroke="${INK}" stroke-width="2.4" stroke-linejoin="miter"` : ""}/>`;
  }).join("\n  ");
}

// Cursor scaled to sit inside the inner ring, with its own hard shadow. Placed by the arrow's
// visual centre (about 8, 12 in path units), not its bounding box, which sits too far left and up.
const ARROW = "M2 2v20l5-4.8 3.6 7.6 3.8-1.8-3.6-7.4H18z";
function cursor(fill, scale, tx, ty) {
  return `<g transform="translate(${tx} ${ty}) scale(${scale})" stroke-linejoin="miter" stroke-miterlimit="10">
    <path d="${ARROW}" fill="${INK}" transform="translate(1.4 1.4)"/>
    <path d="${ARROW}" fill="${fill}" stroke="${INK}" stroke-width="${(2.6 / scale * 1.25).toFixed(2)}"/>
  </g>`;
}

function mark({ bg, outer, inner, innerOutline, cursor: c }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <rect x="10" y="10" width="86" height="86" rx="12" fill="${INK}"/>
  <rect x="4" y="4" width="86" height="86" rx="12" fill="${bg}" stroke="${INK}" stroke-width="5"/>
  ${ring(33, 7, 6, 16, 8, outer, false)}
  ${ring(21.5, 5, 5, 24, 30, inner, innerOutline)}
  ${cursor(c, 1.1, 47 - 8 * 1.1, 47 - 12 * 1.1)}
</svg>
`;
}

// Favicon: below ~40px two rings and a shadow blur together, so keep one ring and a bigger cursor.
function favicon({ bg, outer, cursor: c }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <rect x="3" y="3" width="94" height="94" rx="14" fill="${bg}" stroke="${INK}" stroke-width="6"/>
  ${ring(32, 11, 5, 26, 18, outer, false)}
  ${cursor(c, 1.6, 50 - 8 * 1.6, 50 - 12 * 1.6)}
</svg>
`;
}

const variants = {
  mint: { bg: "#5dcaa5", outer: INK, inner: INK, innerOutline: false, cursor: "#ffffff" },
  sun: { bg: "#ffd23f", outer: INK, inner: INK, innerOutline: false, cursor: "#ffffff" },
  ink: { bg: INK, outer: "#1d9e75", inner: "#5dcaa5", innerOutline: false, cursor: "#ffffff" },
};

for (const [name, v] of Object.entries(variants)) {
  writeFileSync(path.join(dir, `mark-${name}.svg`), mark(v));
  writeFileSync(path.join(dir, `favicon-${name}.svg`), favicon(v));
}

const sizes = [256, 96, 48];
const sheet = `<!doctype html><html><head><meta charset="utf-8"><title>TypeID mark</title><style>
@font-face { font-family: "DM Sans"; font-weight: 500; src: url(../fonts/dm-sans-latin-500-normal.woff2); }
@font-face { font-family: "DM Sans"; font-weight: 700; src: url(../fonts/dm-sans-latin-700-normal.woff2); }
@font-face { font-family: "Space Grotesk"; font-weight: 700; src: url(../fonts/space-grotesk-latin-700-normal.woff2); }
body { margin: 0; padding: 36px 40px 44px; background: #e7f6ef; color: ${INK}; font: 500 14px "DM Sans", sans-serif;
  background-image: linear-gradient(rgba(22,19,40,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(22,19,40,.06) 1px, transparent 1px);
  background-size: 32px 32px; }
h1 { font: 700 30px "Space Grotesk"; margin: 0 0 6px; } h2 { font: 700 18px "Space Grotesk"; margin: 0 0 14px; }
.row { display: flex; align-items: flex-end; gap: 28px; }
.card { background: #fff; border: 2px solid ${INK}; border-radius: 5px; box-shadow: 6px 6px 0 ${INK}; padding: 22px 24px; margin-top: 26px; }
.lbl { font: 700 11.5px "DM Sans"; letter-spacing: .08em; text-transform: uppercase; color: #4b4763; margin-top: 8px; text-align: center; }
.old { display: flex; gap: 20px; align-items: center; }
.bar { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border: 2px solid ${INK}; border-radius: 5px; background: #fff; }
.bar .n { font: 700 24px/1 "Space Grotesk"; } .bar .t { font-size: 13px; color: #4b4763; margin-top: 3px; }
.tab { display: flex; align-items: center; gap: 8px; width: 240px; height: 34px; padding: 0 10px; border: 2px solid ${INK}; border-bottom: 0; border-radius: 8px 8px 0 0; background: #f1f0f6; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; }
</style></head><body>
<h1>TypeID mark, neobrutalist</h1><p>Same idea as the original: two broken scan rings around a cursor. Flat fill, ink outline, hard shadow, square ends. 32 and 16 px use the simplified favicon.</p>
<div class="grid">${Object.keys(variants).map((name) => `
  <div class="card"><h2>${name === "mint" ? "Mint (recommended)" : name === "sun" ? "Sun" : "Ink (closest to the original)"}</h2>
    <div class="row">${sizes.map((s) => `<div><img src="mark-${name}.svg" width="${s}" height="${s}" alt=""><div class="lbl">${s}</div></div>`).join("")}
      ${[32, 16].map((s) => `<div><img src="favicon-${name}.svg" width="${s}" height="${s}" alt=""><div class="lbl">${s} fav</div></div>`).join("")}</div>
    <div style="margin-top:22px" class="bar"><img src="mark-${name}.svg" width="50" height="50" alt=""><div><div class="n">TypeID</div><div class="t">Recognizes you by <b>how</b> you type</div></div></div>
    <div style="margin-top:14px" class="tab"><img src="favicon-${name}.svg" width="16" height="16" alt="">TypeID, identification by typing</div>
  </div>`).join("")}
</div>
<div class="card"><h2>Original, for comparison</h2><div class="old"><img src="../../public/mark.png" width="128" alt=""><img src="../../public/mark.png" width="48" alt=""><img src="../../public/mark.png" width="16" alt=""></div></div>
</body></html>`;
writeFileSync(path.join(dir, "sheet.html"), sheet);
console.log("wrote mark-*.svg, favicon-*.svg and sheet.html");
