from __future__ import annotations

import html
import json
import sys
import typing as tp

from mausoleo.paths import GT_DIR
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Ground truth review</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f4f1e8;
    --surface: #fdfcf7;
    --surface-2: #efebdf;
    --border: #ddd6c5;
    --border-strong: #c6bda6;
    --ink: #262319;
    --muted: #867d68;
    --faint: #a99f88;
    --accent: #33678c;
    --accent-soft: rgba(51,103,140,.14);
    --sepia: #9c6c2e;
    --err: #b1403a;
    --head-bg: #21201a;
    --head-ink: #efe9da;
    --head-muted: #9d9480;
    --head-line: rgba(255,255,255,.14);
    --hl-bg: #fdf6dd;
    --pane-bg: #26241e;
    --shadow: 0 1px 2px rgba(38,30,15,.07), 0 6px 18px rgba(38,30,15,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #171612;
      --surface: #201e18;
      --surface-2: #282520;
      --border: #363228;
      --border-strong: #4c4636;
      --ink: #e5dfcf;
      --muted: #968d78;
      --faint: #6d6655;
      --accent: #74a9ca;
      --accent-soft: rgba(116,169,202,.16);
      --sepia: #c89a5a;
      --err: #de7a71;
      --head-bg: #100f0c;
      --head-ink: #e7e1d1;
      --head-muted: #877f6c;
      --head-line: rgba(255,255,255,.1);
      --hl-bg: #2c2818;
      --pane-bg: #0d0c0a;
      --shadow: 0 1px 2px rgba(0,0,0,.5), 0 6px 18px rgba(0,0,0,.35);
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 20; background: var(--head-bg); color: var(--head-ink);
           padding: .55em 1em; display: flex; gap: .8em; align-items: center; flex-wrap: wrap;
           border-bottom: 1px solid var(--head-line); box-shadow: 0 1px 6px rgba(0,0,0,.25); }
  header a { color: var(--head-ink); opacity: .72; text-decoration: none; }
  header a:hover { opacity: 1; }
  header b { letter-spacing: .02em; font-variant-numeric: tabular-nums; }
  header button, header select {
    font: inherit; font-size: .84em; color: var(--head-ink);
    background: rgba(255,255,255,.07); border: 1px solid var(--head-line);
    border-radius: 6px; padding: .28em .65em; cursor: pointer; transition: background .12s;
  }
  header button:hover, header select:hover { background: rgba(255,255,255,.16); }
  header label { display: inline-flex; gap: .4em; align-items: center; font-size: .85em;
                 opacity: .85; cursor: pointer; }
  header input[type=checkbox] { accent-color: var(--accent); }
  #mdlink { font-size: .85em; text-transform: uppercase; letter-spacing: .06em; }
  #unitcount { font-size: .8em; color: var(--head-muted); font-variant-numeric: tabular-nums; }
  .status { margin-left: auto; font: 12px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
            padding: .3em .75em; border-radius: 99px; background: rgba(255,255,255,.06);
            border: 1px solid var(--head-line); }
  .saved { color: #7fdc9a; } .dirty { color: #ffce6b; } .error { color: #ff8c85; }
  main { display: flex; align-items: flex-start; }
  #imgpane { width: 55%; position: sticky; top: 3.4em; height: calc(100vh - 3.4em); background: var(--pane-bg); }
  #viewport { width: 100%; height: 100%; overflow: auto; position: relative; cursor: crosshair; }
  #viewport.panning { cursor: grabbing; }
  #viewport.canpan { cursor: grab; }
  #canvas { position: relative; width: 100%; }
  #pageimg { width: 100%; display: block; user-select: none; -webkit-user-drag: none; }
  #marquee { position: absolute; border: 1.5px solid #58a6e8; background: rgba(88,166,232,.16);
             display: none; pointer-events: none; z-index: 5; }
  #hud { position: absolute; left: 10px; bottom: 10px; z-index: 10; display: flex; gap: .3em; align-items: center;
         background: rgba(12,11,9,.72); color: #eee; padding: .3em .45em; border-radius: 8px;
         font: 12px ui-monospace, Menlo, Consolas, monospace; backdrop-filter: blur(4px); }
  #hud button { font: inherit; cursor: pointer; border: 0; border-radius: 5px; padding: .2em .6em;
                background: rgba(255,255,255,.12); color: #fff; transition: background .12s; }
  #hud button:hover { background: rgba(255,255,255,.26); }
  #hint { position: absolute; right: 10px; top: 10px; z-index: 10; background: rgba(12,11,9,.62); color: #ddd;
          padding: .4em .65em; border-radius: 8px; font: 11px/1.4 system-ui, sans-serif; max-width: 17em;
          backdrop-filter: blur(4px); }
  #units { width: 45%; padding: 1em 1.2em 6em; }
  .unit { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent);
          border-radius: 8px; margin-bottom: .9em; padding: .7em .9em; box-shadow: var(--shadow);
          cursor: grab; transition: background .12s, border-color .12s; }
  .unit.collapsed { border-left: 3px solid var(--border); box-shadow: none; margin-bottom: .3em;
                    padding: .35em .65em; display: flex; gap: .55em; align-items: baseline; }
  .unit.collapsed:hover { background: var(--surface-2); border-color: var(--border-strong); }
  .unit.multi { border-left: 3px solid var(--sepia); }
  .bar { display: flex; gap: .45em; align-items: center; font-size: .78em; color: var(--muted); margin-bottom: .5em; }
  .bar .id { font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--faint); }
  .bar button { border: 1px solid var(--border); background: var(--surface-2); color: var(--muted);
                border-radius: 5px; cursor: pointer; padding: .1em .55em; font-size: 1em;
                transition: background .12s, color .12s, border-color .12s; }
  .bar button:hover { background: var(--border); color: var(--ink); }
  .bar .del:hover { background: var(--err); border-color: var(--err); color: #fff; }
  input.pages { width: 4.5em; font-size: 1em; font-family: ui-monospace, Menlo, monospace;
                background: var(--surface); color: var(--ink); border: 1px solid var(--border);
                border-radius: 5px; padding: .1em .35em; }
  textarea { width: 100%; border: 1px solid var(--border); border-radius: 6px;
             background: var(--surface); color: var(--ink);
             font: .95em/1.5 Georgia, "Iowan Old Style", "Times New Roman", serif;
             padding: .5em .65em; resize: none; overflow: hidden; }
  textarea:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
  textarea.hl { font-weight: 700; background: var(--hl-bg); margin-bottom: .45em; }
  .unit.collapsed .bar { align-self: flex-start; }
  .unit.collapsed > textarea:not(.hl) { display: none; }
  .unit.collapsed .bar { margin: 0; flex: 0 0 auto; }
  .unit.collapsed .bar button, .unit.collapsed .pages { display: none; }
  .unit.collapsed textarea.hl { display: none; }
  .hltext { font-weight: 600; flex: 0 1 auto; min-width: 0; overflow-wrap: anywhere; }
  .hltext.empty { font-weight: 400; color: var(--err); font-style: italic; }
  .unit:not(.collapsed) .hltext { display: none; }
  .chars { color: var(--faint); font: 11px ui-monospace, Menlo, monospace; flex: 0 0 auto;
           font-variant-numeric: tabular-nums; }
  .snip { color: var(--muted); font-size: .84em; flex: 1 1 0; overflow: hidden; text-overflow: ellipsis;
          white-space: nowrap; min-width: 4em; }
  .unit:not(.collapsed) .snip { display: none; }
  .unit.offpage { opacity: .45; }
  .unit.dragging { opacity: .35; }
  .unit.dropbefore { box-shadow: 0 -3px 0 var(--accent); }
  .unit.dropafter { box-shadow: 0 3px 0 var(--accent); }
  .unit textarea, .unit input { cursor: auto; }
  .addrow { text-align: center; margin: 1.4em 0; }
  .addrow button { font: inherit; font-size: .88em; color: var(--accent); background: none;
                   border: 1px dashed var(--border-strong); border-radius: 8px; padding: .5em 1.1em;
                   cursor: pointer; transition: background .12s; }
  .addrow button:hover { background: var(--accent-soft); }
</style>
<header>
  <a href="/">&#8592; issues</a>
  <b id="title"></b>
  <span>page <select id="pagesel"></select> / <span id="pagecount"></span></span>
  <button onclick="shiftPage(-1)">&#8592; prev</button>
  <button onclick="shiftPage(1)">next &#8594;</button>
  <label><input type="checkbox" id="allpages" onchange="renderUnits()"> whole issue</label>
  <a id="mdlink" href="#">markdown</a>
  <span id="unitcount"></span>
  <span class="status" id="status">loading&#8230;</span>
</header>
<main>
  <div id="imgpane">
    <div id="viewport">
      <div id="canvas"><img id="pageimg"><div id="marquee"></div></div>
    </div>
    <div id="hud">
      <button onclick="zoomStep(1/1.4)">&#8722;</button>
      <span id="zoomlbl">100%</span>
      <button onclick="zoomStep(1.4)">+</button>
      <button onclick="popView()" title="step back out (Esc)">&#8617; back</button>
      <button onclick="resetView()" title="full page">reset</button>
      <span id="depthlbl"></span>
    </div>
    <div id="hint">Drag a box to zoom in &#183; click (or Esc) to zoom back out &#183; Space+drag or scroll to pan &#183; &#8984;/Ctrl+scroll to fine-zoom</div>
  </div>
  <div id="units"></div>
</main>
<script>
const date = location.pathname.split("/").pop();
let doc = null, page = 1, saveTimer = null;

document.getElementById("mdlink").href = `/markdown/${date}`;

const viewport = document.getElementById("viewport");
const canvas = document.getElementById("canvas");
const img = document.getElementById("pageimg");
const marquee = document.getElementById("marquee");
const MAXZOOM = 8;
let stack = [{ zoom: 1, cx: 0.5, cy: 0 }];

function setStatus(cls, text) { const s = document.getElementById("status"); s.className = "status " + cls; s.textContent = text; }

/* ---------- zoom / pan viewer ---------- */
function paneW() { return viewport.clientWidth; }
function curView() { return stack[stack.length - 1]; }

function applyView() {
  const f = curView();
  const w = Math.max(paneW(), paneW() * f.zoom);
  canvas.style.width = w + "px";
  const rh = canvas.clientHeight || 1;
  const maxL = Math.max(0, w - viewport.clientWidth);
  const maxT = Math.max(0, rh - viewport.clientHeight);
  viewport.scrollLeft = Math.min(maxL, Math.max(0, f.cx * w - viewport.clientWidth / 2));
  viewport.scrollTop = Math.min(maxT, Math.max(0, f.cy * rh - viewport.clientHeight / 2));
  document.getElementById("zoomlbl").textContent = Math.round(f.zoom * 100) + "%";
  document.getElementById("depthlbl").textContent = stack.length > 1 ? "\\u25a2\\u00d7" + (stack.length - 1) : "";
  viewport.classList.toggle("canpan", f.zoom > 1.001);
}

function centerFractions() {
  const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  return {
    cx: (viewport.scrollLeft + viewport.clientWidth / 2) / w,
    cy: (viewport.scrollTop + viewport.clientHeight / 2) / h,
  };
}

function resetView() { stack = [{ zoom: 1, cx: 0.5, cy: 0 }]; applyView(); }
function popView() { if (stack.length > 1) { stack.pop(); applyView(); } else resetView(); }

function pushBox(fx0, fy0, fx1, fy1) {
  const bw = Math.max(0.01, fx1 - fx0), bh = Math.max(0.01, fy1 - fy0);
  const aspect = (canvas.clientHeight || 1) / (canvas.clientWidth || 1);
  const zW = 1 / bw;
  const zH = viewport.clientHeight / (bh * paneW() * aspect);
  const zoom = Math.max(1, Math.min(MAXZOOM, Math.min(zW, zH)));
  stack.push({ zoom, cx: (fx0 + fx1) / 2, cy: (fy0 + fy1) / 2 });
  applyView();
}

function zoomStep(factor) {
  const f = curView();
  const nz = Math.max(1, Math.min(MAXZOOM, f.zoom * factor));
  const c = centerFractions();
  if (Math.abs(nz - f.zoom) < 1e-3) return;
  if (f.zoom === 1 && nz > 1) stack.push({ zoom: nz, cx: c.cx, cy: c.cy });
  else { f.zoom = nz; f.cx = c.cx; f.cy = c.cy; }
  applyView();
}

function zoomAt(factor, clientX, clientY) {
  const f = curView();
  const nz = Math.max(1, Math.min(MAXZOOM, f.zoom * factor));
  if (Math.abs(nz - f.zoom) < 1e-3) return;
  const r = viewport.getBoundingClientRect();
  const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  const fracX = (viewport.scrollLeft + (clientX - r.left)) / w;
  const fracY = (viewport.scrollTop + (clientY - r.top)) / h;
  const target = (f.zoom === 1 && nz > 1) ? { zoom: nz, cx: fracX, cy: fracY } : f;
  target.zoom = nz;
  if (f.zoom === 1 && nz > 1) stack.push(target);
  const w2 = paneW() * nz, h2 = w2 * ((canvas.clientHeight || 1) / (canvas.clientWidth || 1));
  const sl = fracX * w2 - (clientX - r.left);
  const st = fracY * h2 - (clientY - r.top);
  target.cx = (sl + viewport.clientWidth / 2) / w2;
  target.cy = (st + viewport.clientHeight / 2) / h2;
  applyView();
}

let spaceDown = false, drag = null;
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "TEXTAREA" && e.target.tagName !== "INPUT") { spaceDown = true; viewport.classList.add("canpan"); e.preventDefault(); }
  if (e.key === "Escape") popView();
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") { spaceDown = false; viewport.classList.remove("canpan"); } });

viewport.addEventListener("contextmenu", (e) => { e.preventDefault(); popView(); });

viewport.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  const r = viewport.getBoundingClientRect();
  if (spaceDown || curView().zoom > 1.001 && e.shiftKey) {
    drag = { mode: "pan", x: e.clientX, y: e.clientY, sl: viewport.scrollLeft, st: viewport.scrollTop };
    viewport.classList.add("panning");
  } else {
    drag = { mode: "box", x0: e.clientX, y0: e.clientY };
  }
  e.preventDefault();
});

window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  if (drag.mode === "pan") {
    viewport.scrollLeft = drag.sl - (e.clientX - drag.x);
    viewport.scrollTop = drag.st - (e.clientY - drag.y);
    return;
  }
  const cr = canvas.getBoundingClientRect();
  const x0 = Math.min(drag.x0, e.clientX), y0 = Math.min(drag.y0, e.clientY);
  const x1 = Math.max(drag.x0, e.clientX), y1 = Math.max(drag.y0, e.clientY);
  marquee.style.display = "block";
  marquee.style.left = (x0 - cr.left) + "px";
  marquee.style.top = (y0 - cr.top) + "px";
  marquee.style.width = (x1 - x0) + "px";
  marquee.style.height = (y1 - y0) + "px";
});

window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  if (drag.mode === "pan") {
    const c = centerFractions(); curView().cx = c.cx; curView().cy = c.cy;
    viewport.classList.remove("panning"); drag = null; return;
  }
  marquee.style.display = "none";
  const cr = canvas.getBoundingClientRect();
  const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  const dx = Math.abs(e.clientX - drag.x0), dy = Math.abs(e.clientY - drag.y0);
  if (dx > 8 && dy > 8) {
    const fx0 = (Math.min(drag.x0, e.clientX) - cr.left) / w;
    const fy0 = (Math.min(drag.y0, e.clientY) - cr.top) / h;
    const fx1 = (Math.max(drag.x0, e.clientX) - cr.left) / w;
    const fy1 = (Math.max(drag.y0, e.clientY) - cr.top) / h;
    pushBox(Math.max(0, fx0), Math.max(0, fy0), Math.min(1, fx1), Math.min(1, fy1));
  } else if (dx < 6 && dy < 6 && stack.length > 1) {
    popView();
  }
  drag = null;
});

viewport.addEventListener("wheel", (e) => {
  if (e.ctrlKey || e.metaKey) { e.preventDefault(); zoomAt(e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX, e.clientY); }
}, { passive: false });

window.addEventListener("resize", applyView);

/* ---------- data / units ---------- */
async function load() {
  doc = await (await fetch(`/api/${date}`)).json();
  document.getElementById("title").textContent = date;
  document.getElementById("pagecount").textContent = doc.page_count;
  const sel = document.getElementById("pagesel");
  sel.innerHTML = "";
  for (let p = 1; p <= doc.page_count; p++) sel.append(new Option(p, p));
  sel.onchange = () => { page = +sel.value; showPage(); };
  showPage();
  setStatus("saved", "loaded");
}

function showPage() {
  document.getElementById("pagesel").value = page;
  img.onload = () => { resetView(); };
  img.src = `/img/${date}/${page}`;
  renderUnits();
  window.scrollTo(0, 0);
}

function shiftPage(d) { page = Math.min(doc.page_count, Math.max(1, page + d)); showPage(); }

function markDirty() { setStatus("dirty", "unsaved\\u2026"); clearTimeout(saveTimer); saveTimer = setTimeout(save, 700); }

async function save() {
  setStatus("dirty", "saving\\u2026");
  const res = await fetch(`/api/${date}`, { method: "POST", body: JSON.stringify(doc) });
  if (res.ok) { doc = await res.json(); setStatus("saved", "saved \\u2713"); renderUnits(); }
  else setStatus("error", "SAVE FAILED");
}

function autosize(t) { t.style.height = "auto"; t.style.height = t.scrollHeight + 2 + "px"; }
function btn(label, fn) { const b = document.createElement("button"); b.textContent = label; b.onclick = fn; return b; }

function visibleIndices() {
  return doc.articles.map((a, i) => [a, i]).filter(([a]) => a.page_span.includes(page)).map(([, i]) => i);
}

let dragFrom = null;

function clearDropMarks() {
  for (const el of document.querySelectorAll(".dropbefore, .dropafter")) el.classList.remove("dropbefore", "dropafter");
}

function dropOn(targetGi, after) {
  if (dragFrom === null || dragFrom === targetGi) return;
  const from = dragFrom;
  dragFrom = null;
  const [a] = doc.articles.splice(from, 1);
  let at = targetGi + (after ? 1 : 0);
  if (from < at) at -= 1;
  doc.articles.splice(at, 0, a);
  markDirty(); renderUnits();
}

function preview(text) {
  const flat = text.replace(/\\s+/g, " ").trim();
  if (!flat) return "";
  const m = flat.match(/^.*?[.!?](?=\\s|$)(?:\\s+.*?[.!?](?=\\s|$))?/);
  const out = m ? m[0] : flat;
  return out.length > 180 ? out.slice(0, 180) + "\\u2026" : out;
}

function unitEl(art, gi) {
  const d = document.createElement("div");
  const hl = document.createElement("textarea");
  const body = document.createElement("textarea");
  d.className = "unit collapsed" + (art.page_span.length > 1 ? " multi" : "");
  d.draggable = true;
  d.dataset.gi = gi;
  const setOpen = open => {
    d.classList.toggle("collapsed", !open);
    if (open) requestAnimationFrame(() => { autosize(hl); autosize(body); });
  };
  d.onclick = ev => {
    if (ev.target.closest("textarea, input, button")) return;
    setOpen(d.classList.contains("collapsed"));
  };
  d.ondragstart = ev => { dragFrom = gi; d.classList.add("dragging"); ev.dataTransfer.effectAllowed = "move"; };
  d.ondragend = () => { d.classList.remove("dragging"); clearDropMarks(); };
  d.ondragover = ev => {
    if (dragFrom === null || dragFrom === gi) return;
    ev.preventDefault();
    const r = d.getBoundingClientRect();
    clearDropMarks();
    d.classList.add(ev.clientY < r.top + r.height / 2 ? "dropbefore" : "dropafter");
  };
  d.ondragleave = () => d.classList.remove("dropbefore", "dropafter");
  d.ondrop = ev => {
    ev.preventDefault();
    const after = d.classList.contains("dropafter");
    clearDropMarks();
    dropOn(gi, after);
  };
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.insertAdjacentHTML("beforeend", `<span class="id">${art.id.split("_").pop()}</span>`);
  const hltext = document.createElement("span");
  const setHeadline = () => {
    hltext.textContent = art.headline || "(no headline)";
    hltext.className = "hltext" + (art.headline ? "" : " empty");
  };
  setHeadline();
  const snip = document.createElement("span");
  snip.className = "snip";
  snip.textContent = preview(art.paragraphs.map(p => p.text).join(" "));
  const chars = document.createElement("span");
  chars.className = "chars";
  chars.textContent = `p${art.page_span.join(",")} \u00b7 ${art.paragraphs.reduce((n,p)=>n+p.text.length,0)}c`;
  const pages = document.createElement("input");
  pages.className = "pages";
  pages.value = art.page_span.join(",");
  pages.onchange = () => {
    const ps = pages.value.split(",").map(x => parseInt(x.trim())).filter(x => x >= 1 && x <= doc.page_count);
    if (ps.length) { art.page_span = [...new Set(ps)].sort((a,b)=>a-b); markDirty(); renderUnits(); }
  };
  const del = btn("\\u2715", () => { if (confirm("Delete this unit?")) { doc.articles.splice(gi, 1); markDirty(); renderUnits(); } });
  del.className = "del";
  bar.append(pages, del);
  d.append(bar);

  hl.className = "hl"; hl.placeholder = "(no headline)"; hl.value = art.headline || ""; hl.rows = 1;
  hl.oninput = () => { art.headline = hl.value.trim() ? hl.value : null; setHeadline(); autosize(hl); markDirty(); };
  d.append(hltext, hl);

  body.value = art.paragraphs.map(p => p.text).join("\\n\\n");
  body.oninput = () => {
    art.paragraphs = body.value.split(/\\n\\s*\\n/).map(t => ({ text: t.trim() })).filter(p => p.text);
    snip.textContent = preview(body.value);
    chars.textContent = `p${art.page_span.join(",")} \\u00b7 ${body.value.length}c`;
    autosize(body); markDirty();
  };
  d.append(body, snip, chars);
  return d;
}

function renderUnits() {
  const box = document.getElementById("units");
  box.innerHTML = "";
  const whole = document.getElementById("allpages").checked;
  const onPage = new Set(visibleIndices());
  const idxs = whole ? doc.articles.map((_, i) => i) : [...onPage];
  document.getElementById("unitcount").textContent = `${onPage.size} on page \\u00b7 ${doc.articles.length} total`;
  for (const gi of idxs) {
    const el = unitEl(doc.articles[gi], gi);
    if (!onPage.has(gi)) el.classList.add("offpage");
    box.append(el);
  }
  const addRow = document.createElement("div");
  addRow.className = "addrow";
  addRow.append(btn("+ add unit on this page", () => {
    const idxs2 = visibleIndices();
    const at = idxs2.length ? idxs2[idxs2.length - 1] + 1 : doc.articles.length;
    doc.articles.splice(at, 0, { id: "new", headline: null, paragraphs: [], page_span: [page] });
    markDirty(); renderUnits();
  }));
  box.append(addRow);
}

load();
</script>
"""

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Ground truth review</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f4f1e8; --surface: #fdfcf7; --border: #ddd6c5; --ink: #262319;
    --muted: #867d68; --accent: #33678c; --accent-soft: rgba(51,103,140,.12);
    --shadow: 0 1px 2px rgba(38,30,15,.07), 0 6px 18px rgba(38,30,15,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #171612; --surface: #201e18; --border: #363228; --ink: #e5dfcf;
      --muted: #968d78; --accent: #74a9ca; --accent-soft: rgba(116,169,202,.16);
      --shadow: 0 1px 2px rgba(0,0,0,.5), 0 6px 18px rgba(0,0,0,.35);
    }
  }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 34em; margin: 4em auto 6em; padding: 0 1.2em; }
  h1 { font-size: 1.5em; letter-spacing: .01em; margin: 0 0 .2em; }
  .sub { color: var(--muted); font-size: .9em; margin: 0 0 1.8em; }
  ul { list-style: none; margin: 0; padding: 0; }
  li { display: flex; align-items: stretch; background: var(--surface); border: 1px solid var(--border);
       border-radius: 10px; margin: .5em 0; overflow: hidden;
       transition: box-shadow .12s, border-color .12s; }
  li:hover { border-color: var(--accent); box-shadow: var(--shadow); }
  a.issue { flex: 1; padding: .75em 1em; text-decoration: none; color: var(--ink);
            font-size: 1.08em; font-weight: 500; font-variant-numeric: tabular-nums; }
  a.md { display: flex; align-items: center; padding: 0 1.1em; text-decoration: none; color: var(--accent);
         font-size: .78em; text-transform: uppercase; letter-spacing: .07em;
         border-left: 1px solid var(--border); }
  a.md:hover { background: var(--accent-soft); }
</style>
<main>
  <h1>Ground truth review</h1>
  <p class="sub">Il Messaggero &#183; {count} issues</p>
  <ul>{links}</ul>
</main>
"""

MARKDOWN_HTML = """<!doctype html>
<meta charset="utf-8">
<title>{date} &#183; markdown</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f4f1e8; --border: #ddd6c5; --ink: #262319; --muted: #867d68;
    --head-bg: #21201a; --head-ink: #efe9da; --head-line: rgba(255,255,255,.14);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #171612; --border: #363228; --ink: #e5dfcf; --muted: #968d78;
      --head-bg: #100f0c; --head-ink: #e7e1d1; --head-line: rgba(255,255,255,.1);
    }
  }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.6 Georgia, "Iowan Old Style", "Times New Roman", serif; }
  header { position: sticky; top: 0; z-index: 10; background: var(--head-bg); color: var(--head-ink);
           display: flex; gap: 1em; align-items: center; padding: .55em 1.2em;
           font: 14px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
           border-bottom: 1px solid var(--head-line); }
  header a { color: inherit; opacity: .72; text-decoration: none; }
  header a:hover { opacity: 1; }
  header b { font-variant-numeric: tabular-nums; }
  header .spacer { flex: 1; }
  header button { font: inherit; font-size: .9em; color: var(--head-ink);
                  background: rgba(255,255,255,.07); border: 1px solid var(--head-line);
                  border-radius: 6px; padding: .28em .75em; cursor: pointer; transition: background .12s; }
  header button:hover { background: rgba(255,255,255,.16); }
  main { max-width: 44em; margin: 2.5em auto 6em; padding: 0 1.3em; }
  h1 { font-size: 1.7em; line-height: 1.25; margin: 0 0 .15em; }
  .meta { color: var(--muted); font-style: italic; margin: 0 0 2em; font-size: .95em; }
  .pagemark { display: flex; align-items: center; gap: 1em; margin: 2.6em 0 1.6em; color: var(--muted);
              font: 12px system-ui, sans-serif; text-transform: uppercase; letter-spacing: .14em; }
  .pagemark::before, .pagemark::after { content: ""; flex: 1; border-top: 1px solid var(--border); }
  article { margin: 0 0 2.2em; }
  h3 { font-size: 1.15em; line-height: 1.35; margin: 0 0 .5em; }
  h3 .pp { font-weight: 400; font-style: italic; color: var(--muted); font-size: .8em; margin-left: .6em; }
  h3.untitled { font-weight: 400; font-style: italic; color: var(--muted); }
  p { margin: 0 0 .85em; }
</style>
<header>
  <a href="/">&#8592; issues</a>
  <a href="/issue/{date}">review</a>
  <b>{date}</b>
  <span class="spacer"></span>
  <button id="copybtn" onclick="copyMarkdown()">copy markdown</button>
  <a href="/markdown/{date}/raw">view raw</a>
</header>
<main>
  <h1>Il Messaggero &#8212; {date}</h1>
  <p class="meta">{meta}</p>
{body}
</main>
<script>
async function copyMarkdown() {
  const btn = document.getElementById("copybtn");
  try {
    const text = await (await fetch("/markdown/{date}/raw")).text();
    await navigator.clipboard.writeText(text);
    btn.textContent = "copied \u2713";
  } catch { btn.textContent = "copy failed"; }
  setTimeout(() => { btn.textContent = "copy markdown"; }, 1600);
}
</script>
"""


def issue_dates() -> list[str]:
    return sorted(p.name for p in GT_DIR.iterdir() if (p / "ground_truth.json").exists())


def load_issue(date: str) -> dict[str, tp.Any]:
    return json.loads((GT_DIR / date / "ground_truth.json").read_text())


def ordered_articles(issue: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
    return sorted(issue["articles"], key=lambda a: a.get("position_in_issue", 0))


def _page_span_label(art: dict[str, tp.Any]) -> str:
    span = art.get("page_span") or []
    return ", ".join(str(p) for p in span) if len(span) > 1 else ""


def _article_markdown(art: dict[str, tp.Any]) -> list[str]:
    span_label = _page_span_label(art)
    suffix = f" *(pp. {span_label})*" if span_label else ""
    title = art.get("headline") or "*Senza titolo*"
    lines = [f"### {title}{suffix}", ""]
    for para in art.get("paragraphs", []):
        lines += [para["text"], ""]
    return lines


def render_issue_markdown(issue: dict[str, tp.Any]) -> str:
    articles = ordered_articles(issue)
    meta = f"*{issue['page_count']} pages · {len(articles)} articles*"
    lines = [f"# Il Messaggero — {issue['date']}", "", meta, ""]
    current_page = 0
    for art in articles:
        first_page = min(art.get("page_span") or [1])
        if first_page != current_page:
            current_page = first_page
            lines += ["---", "", f"## Page {current_page}", ""]
        lines += _article_markdown(art)
    return "\n".join(lines).rstrip() + "\n"


def _article_html(art: dict[str, tp.Any]) -> str:
    span_label = _page_span_label(art)
    pp = f'<span class="pp">pp. {span_label}</span>' if span_label else ""
    headline = art.get("headline")
    heading = f"<h3>{html.escape(headline)}{pp}</h3>" if headline else f'<h3 class="untitled">Senza titolo{pp}</h3>'
    paragraphs = "\n".join(f"<p>{html.escape(p['text'])}</p>" for p in art.get("paragraphs", []))
    return f"<article>{heading}\n{paragraphs}</article>"


def render_markdown_page(issue: dict[str, tp.Any]) -> str:
    articles = ordered_articles(issue)
    parts: list[str] = []
    current_page = 0
    for art in articles:
        first_page = min(art.get("page_span") or [1])
        if first_page != current_page:
            current_page = first_page
            parts.append(f'<div class="pagemark"><span>Page {current_page}</span></div>')
        parts.append(_article_html(art))
    meta = f"{issue['page_count']} pages · {len(articles)} articles"
    page = MARKDOWN_HTML.replace("{date}", html.escape(issue["date"])).replace("{meta}", meta)
    return page.replace("{body}", "\n".join(parts))


def render_index() -> str:
    dates = issue_dates()
    links = "\n".join(f'<li><a class="issue" href="/issue/{d}">{d}</a><a class="md" href="/markdown/{d}">markdown</a></li>' for d in dates)
    return INDEX_HTML.replace("{count}", str(len(dates))).replace("{links}", links)


def renumber(issue: dict[str, tp.Any]) -> dict[str, tp.Any]:
    date = issue["date"]
    for idx, art in enumerate(issue["articles"]):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        art["page_span"] = sorted(set(art.get("page_span") or [1]))
        for p_idx, para in enumerate(art.get("paragraphs", [])):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    return issue


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if not parts:
            self._send(200, render_index().encode(), "text/html; charset=utf-8")
        elif parts[0] == "issue" and len(parts) == 2:
            self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
        elif parts[0] == "api" and len(parts) == 2:
            path = GT_DIR / parts[1] / "ground_truth.json"
            self._send(200, path.read_bytes(), "application/json")
        elif parts[0] == "markdown" and len(parts) == 2:
            self._send(200, render_markdown_page(load_issue(parts[1])).encode(), "text/html; charset=utf-8")
        elif parts[0] == "markdown" and len(parts) == 3 and parts[2] == "raw":
            self._send(200, render_issue_markdown(load_issue(parts[1])).encode(), "text/plain; charset=utf-8")
        elif parts == ["favicon.ico"]:
            self._send(204, b"", "image/x-icon")
        elif parts[0] == "img" and len(parts) == 3:
            path = GT_DIR / parts[1] / f"{parts[2]}.jpeg"
            self._send(200, path.read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if parts[0] != "api" or len(parts) != 2 or parts[1] not in issue_dates():
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers["Content-Length"])
        issue = renumber(json.loads(self.rfile.read(length)))
        path = GT_DIR / parts[1] / "ground_truth.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
        tmp.replace(path)
        self._send(200, json.dumps(issue, ensure_ascii=False).encode(), "application/json")

    def log_message(self, fmt: str, *args: tp.Any) -> None:
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"review server on http://0.0.0.0:{port} — issues: {', '.join(issue_dates())}")
    server.serve_forever()


if __name__ == "__main__":
    main()
