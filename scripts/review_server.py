from __future__ import annotations

import json
import pathlib as pl
import sys
import typing as tp
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TENTATIVE_DIR = pl.Path("eval/tentative_gt")

PAGE_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Tentative GT review</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f5f2ec; }
  header { position: sticky; top: 0; z-index: 10; background: #1f2430; color: #fff; padding: .5em 1em;
           display: flex; gap: 1em; align-items: center; flex-wrap: wrap; }
  header a, header button, header select { font-size: .95em; }
  header .status { margin-left: auto; font-family: monospace; }
  .saved { color: #7fdc9a; } .dirty { color: #ffce6b; } .error { color: #ff7b7b; }
  main { display: flex; align-items: flex-start; }
  #imgpane { width: 55%; position: sticky; top: 3.2em; max-height: calc(100vh - 3.4em); overflow: auto;
             background: #333; text-align: center; }
  #imgpane img { width: 100%; display: block; }
  #units { width: 45%; padding: .8em 1.2em 6em; box-sizing: border-box; }
  .unit { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: .9em; padding: .6em .8em;
          box-shadow: 0 1px 2px rgba(0,0,0,.06); }
  .unit.multi { border-left: 4px solid #7a5cff; }
  .bar { display: flex; gap: .4em; align-items: center; font-size: .8em; color: #666; margin-bottom: .4em; }
  .bar .id { font-family: monospace; }
  .bar button { border: 1px solid #ccc; background: #fafafa; border-radius: 4px; cursor: pointer; padding: 0 .5em; }
  .bar button:hover { background: #eee; }
  .bar .del:hover { background: #ffdddd; }
  select.type { font-size: .8em; }
  input.pages { width: 4.5em; font-size: .8em; font-family: monospace; }
  textarea { width: 100%; box-sizing: border-box; border: 1px solid #e2ddd2; border-radius: 4px;
             font-family: Georgia, serif; font-size: .95em; line-height: 1.35; padding: .35em .5em; resize: none; }
  textarea.hl { font-weight: bold; background: #fffbe8; margin-bottom: .35em; }
  .pagejump { padding: .6em 0; text-align: center; }
  .pagejump button { margin: 0 .2em; }
  .addrow { text-align: center; margin: 1em 0; }
</style>
<header>
  <b id="title"></b>
  <span>page <select id="pagesel"></select> / <span id="pagecount"></span></span>
  <button onclick="shiftPage(-1)">&#8592; prev</button>
  <button onclick="shiftPage(1)">next &#8594;</button>
  <span id="unitcount"></span>
  <span class="status" id="status">loading…</span>
</header>
<main>
  <div id="imgpane"><img id="pageimg"></div>
  <div id="units"></div>
</main>
<script>
const date = location.pathname.split("/").pop();
let doc = null, page = 1, saveTimer = null;

function setStatus(cls, text) { const s = document.getElementById("status"); s.className = "status " + cls; s.textContent = text; }

async function load() {
  doc = await (await fetch(`/api/${date}`)).json();
  document.getElementById("title").textContent = date;
  document.getElementById("pagecount").textContent = doc.page_count;
  const sel = document.getElementById("pagesel");
  sel.innerHTML = "";
  for (let p = 1; p <= doc.page_count; p++) sel.append(new Option(p, p));
  sel.onchange = () => { page = +sel.value; render(); };
  render();
  setStatus("saved", "loaded");
}

function shiftPage(d) { page = Math.min(doc.page_count, Math.max(1, page + d)); document.getElementById("pagesel").value = page; render(); }

function markDirty() {
  setStatus("dirty", "unsaved…");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 700);
}

async function save() {
  setStatus("dirty", "saving…");
  const res = await fetch(`/api/${date}`, { method: "POST", body: JSON.stringify(doc) });
  if (res.ok) { doc = await res.json(); setStatus("saved", "saved ✓"); refreshIds(); }
  else setStatus("error", "SAVE FAILED");
}

function refreshIds() {
  document.querySelectorAll(".unit").forEach((el, i) => {});
  render(false);
}

function autosize(t) { t.style.height = "auto"; t.style.height = t.scrollHeight + 2 + "px"; }

function unitEl(art, gi) {
  const d = document.createElement("div");
  d.className = "unit" + (art.page_span.length > 1 ? " multi" : "");
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<span class="id">${art.id.split("_").pop()}</span>`;
  const type = document.createElement("select");
  type.className = "type";
  for (const t of ["article","advertisement","notice","obituary","editorial","other"]) type.append(new Option(t, t));
  type.value = art.unit_type;
  type.onchange = () => { art.unit_type = type.value; markDirty(); };
  const pages = document.createElement("input");
  pages.className = "pages";
  pages.value = art.page_span.join(",");
  pages.onchange = () => {
    const ps = pages.value.split(",").map(x => parseInt(x.trim())).filter(x => x >= 1 && x <= doc.page_count);
    if (ps.length) { art.page_span = [...new Set(ps)].sort((a,b)=>a-b); markDirty(); render(false); }
  };
  const up = btn("↑", () => move(gi, -1));
  const down = btn("↓", () => move(gi, 1));
  const del = btn("✕ delete", () => { if (confirm("Delete this unit?")) { doc.articles.splice(gi, 1); markDirty(); render(false); } });
  del.className = "del";
  bar.append(type, pages, up, down, del);
  d.append(bar);

  const hl = document.createElement("textarea");
  hl.className = "hl";
  hl.placeholder = "(no headline)";
  hl.value = art.headline || "";
  hl.rows = 1;
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

function btn(label, fn) { const b = document.createElement("button"); b.textContent = label; b.onclick = fn; return b; }

function move(gi, delta) {
  const order = visibleIndices();
  const pos = order.indexOf(gi);
  const target = order[pos + delta];
  if (target === undefined) return;
  const [a] = doc.articles.splice(gi, 1);
  doc.articles.splice(target > gi ? target : target, 0, a);
  markDirty(); render(false);
}

function visibleIndices() {
  return doc.articles.map((a, i) => [a, i]).filter(([a]) => a.page_span.includes(page)).map(([, i]) => i);
}

function render(resetScroll = true) {
  document.getElementById("pageimg").src = `/img/${date}/${page}`;
  const box = document.getElementById("units");
  box.innerHTML = "";
  const idxs = visibleIndices();
  document.getElementById("unitcount").textContent = `${idxs.length} units on page · ${doc.articles.length} total`;
  for (const gi of idxs) box.append(unitEl(doc.articles[gi], gi));
  const addRow = document.createElement("div");
  addRow.className = "addrow";
  addRow.append(btn("+ add unit on this page", () => {
    const last = idxs.length ? idxs[idxs.length - 1] + 1 : doc.articles.length;
    doc.articles.splice(last, 0, { id: "new", unit_type: "article", headline: null, paragraphs: [], page_span: [page], position_in_issue: 0 });
    markDirty(); render(false);
  }));
  box.append(addRow);
  if (resetScroll) window.scrollTo(0, 0);
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
        elif parts[0] == "img" and len(parts) == 3:
            path = TENTATIVE_DIR / parts[1] / "pages" / f"{parts[2]}.jpeg"
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
