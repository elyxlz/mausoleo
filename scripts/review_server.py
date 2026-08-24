from __future__ import annotations

import json
import pathlib as pl
import sys
import typing as tp
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TENTATIVE_DIR = pl.Path("eval/ground_truth")

PAGE_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Tentative GT review</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f5f2ec; }
  header { position: sticky; top: 0; z-index: 20; background: #1f2430; color: #fff; padding: .5em 1em;
           display: flex; gap: 1em; align-items: center; flex-wrap: wrap; }
  header a { color: #9ec7ff; }
  header button, header select { font-size: .95em; }
  header .status { margin-left: auto; font-family: monospace; }
  .saved { color: #7fdc9a; } .dirty { color: #ffce6b; } .error { color: #ff7b7b; }
  main { display: flex; align-items: flex-start; }
  #imgpane { width: 55%; position: sticky; top: 3.2em; height: calc(100vh - 3.4em); background: #2b2b2b; }
  #viewport { width: 100%; height: 100%; overflow: auto; position: relative; cursor: crosshair; }
  #viewport.panning { cursor: grabbing; }
  #viewport.canpan { cursor: grab; }
  #canvas { position: relative; width: 100%; }
  #pageimg { width: 100%; display: block; user-select: none; -webkit-user-drag: none; }
  #marquee { position: absolute; border: 2px solid #4da3ff; background: rgba(77,163,255,.18);
             display: none; pointer-events: none; z-index: 5; }
  #hud { position: absolute; left: 8px; bottom: 8px; z-index: 10; display: flex; gap: .35em; align-items: center;
         background: rgba(0,0,0,.62); color: #fff; padding: .3em .5em; border-radius: 6px; font: 12px monospace; }
  #hud button { font: 12px monospace; cursor: pointer; border: 0; border-radius: 4px; padding: .15em .55em; background: #4a4a4a; color: #fff; }
  #hud button:hover { background: #666; }
  #hint { position: absolute; right: 8px; top: 8px; z-index: 10; background: rgba(0,0,0,.55); color: #eee;
          padding: .35em .6em; border-radius: 6px; font: 11px sans-serif; max-width: 16em; line-height: 1.35; }
  #units { width: 45%; padding: .8em 1.2em 6em; box-sizing: border-box; }
  .unit { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: .9em; padding: .6em .8em;
          box-shadow: 0 1px 2px rgba(0,0,0,.06); }
  .unit.multi { border-left: 4px solid #7a5cff; }
  .bar { display: flex; gap: .4em; align-items: center; font-size: .8em; color: #666; margin-bottom: .4em; }
  .bar .id { font-family: monospace; }
  .bar button { border: 1px solid #ccc; background: #fafafa; border-radius: 4px; cursor: pointer; padding: 0 .5em; }
  .bar button:hover { background: #eee; }
  .bar .del:hover { background: #ffdddd; }
  select.type { font-size: .8em; } input.pages { width: 4.5em; font-size: .8em; font-family: monospace; }
  textarea { width: 100%; box-sizing: border-box; border: 1px solid #e2ddd2; border-radius: 4px;
             font-family: Georgia, serif; font-size: .95em; line-height: 1.35; padding: .35em .5em; resize: none; overflow: hidden; }
  textarea.hl { font-weight: bold; background: #fffbe8; margin-bottom: .35em; }
  .addrow { text-align: center; margin: 1em 0; }
</style>
<header>
  <a href="/">&#8592; issues</a>
  <b id="title"></b>
  <span>page <select id="pagesel"></select> / <span id="pagecount"></span></span>
  <button onclick="shiftPage(-1)">&#8592; prev</button>
  <button onclick="shiftPage(1)">next &#8594;</button>
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

function move(gi, delta) {
  const order = visibleIndices();
  const pos = order.indexOf(gi);
  const target = order[pos + delta];
  if (target === undefined) return;
  const [a] = doc.articles.splice(gi, 1);
  doc.articles.splice(target, 0, a);
  markDirty(); renderUnits();
}

function unitEl(art, gi) {
  const d = document.createElement("div");
  d.className = "unit" + (art.page_span.length > 1 ? " multi" : "");
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<span class="id">${art.id.split("_").pop()}</span>`;
  const pages = document.createElement("input");
  pages.className = "pages";
  pages.value = art.page_span.join(",");
  pages.onchange = () => {
    const ps = pages.value.split(",").map(x => parseInt(x.trim())).filter(x => x >= 1 && x <= doc.page_count);
    if (ps.length) { art.page_span = [...new Set(ps)].sort((a,b)=>a-b); markDirty(); renderUnits(); }
  };
  const del = btn("\\u2715", () => { if (confirm("Delete this unit?")) { doc.articles.splice(gi, 1); markDirty(); renderUnits(); } });
  del.className = "del";
  bar.append(pages, btn("\\u2191", () => move(gi, -1)), btn("\\u2193", () => move(gi, 1)), del);
  d.append(bar);

  const hl = document.createElement("textarea");
  hl.className = "hl"; hl.placeholder = "(no headline)"; hl.value = art.headline || ""; hl.rows = 1;
  hl.oninput = () => { art.headline = hl.value.trim() ? hl.value : null; autosize(hl); markDirty(); };
  d.append(hl);

  const body = document.createElement("textarea");
  body.value = art.paragraphs.map(p => p.text).join("\\n\\n");
  body.oninput = () => {
    art.paragraphs = body.value.split(/\\n\\s*\\n/).map(t => ({ text: t.trim() })).filter(p => p.text);
    autosize(body); markDirty();
  };
  d.append(body);
  requestAnimationFrame(() => { autosize(hl); autosize(body); });
  return d;
}

function renderUnits() {
  const box = document.getElementById("units");
  box.innerHTML = "";
  const idxs = visibleIndices();
  document.getElementById("unitcount").textContent = `${idxs.length} on page \\u00b7 ${doc.articles.length} total`;
  for (const gi of idxs) box.append(unitEl(doc.articles[gi], gi));
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

INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>Tentative GT review</title>
<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:40em;margin:3em auto}
a{display:block;font-size:1.3em;padding:.6em;border:1px solid #ddd;border-radius:6px;margin:.5em 0;text-decoration:none;color:#1f2430}
a:hover{background:#f2efe8}</style>
<h1>Tentative GT review</h1>
{links}
"""


def issue_dates() -> list[str]:
    return sorted(p.name for p in TENTATIVE_DIR.iterdir() if (p / "ground_truth.json").exists())


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
            links = "\n".join(f'<a href="/issue/{d}">{d}</a>' for d in issue_dates())
            self._send(200, INDEX_HTML.replace("{links}", links).encode(), "text/html; charset=utf-8")
        elif parts[0] == "issue" and len(parts) == 2:
            self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
        elif parts[0] == "api" and len(parts) == 2:
            path = TENTATIVE_DIR / parts[1] / "ground_truth.json"
            self._send(200, path.read_bytes(), "application/json")
        elif parts == ["favicon.ico"]:
            self._send(204, b"", "image/x-icon")
        elif parts[0] == "img" and len(parts) == 3:
            path = TENTATIVE_DIR / parts[1] / f"{parts[2]}.jpeg"
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
        path = TENTATIVE_DIR / parts[1] / "ground_truth.json"
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
