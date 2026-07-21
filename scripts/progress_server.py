from __future__ import annotations

import http.server
import json
import pathlib as pl
import socketserver
import typing as tp

LOG = pl.Path("eval/autoresearch/mausoleobench_log.jsonl")
PORT = 8078


def load_attempts() -> list[dict[str, tp.Any]]:
    if not LOG.exists():
        return []
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["n"])


def build_payload() -> dict[str, tp.Any]:
    rows = load_attempts()
    attempts = [r for r in rows if not r.get("reference")]
    references = [r for r in rows if r.get("reference")]
    best = 0.0
    for r in sorted(attempts, key=lambda r: r["n"]):
        r["record"] = r["score"] > best + 1e-9
        if r["record"]:
            best = r["score"]
    return {"attempts": attempts, "references": references, "best": best}


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>MausoleoBench — live progress</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:20px 28px 8px}
h1{margin:0;font-size:22px;letter-spacing:.5px}
.sub{color:#8b949e;font-size:13px;margin-top:4px}
#wrap{padding:8px 28px 40px}
canvas{width:100%;max-width:1100px;background:#0d1117;display:block}
table{border-collapse:collapse;width:100%;max-width:1100px;margin-top:18px;font-size:13px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.6px}
tr.rec td{color:#e6edf3}
.score{font-variant-numeric:tabular-nums;font-weight:700;color:#f2cc60}
.n{color:#8b949e;font-variant-numeric:tabular-nums}
.name{font-family:ui-monospace,Menlo,monospace;color:#7ee787}
.delta{color:#3fb950;font-variant-numeric:tabular-nums;font-size:12px}
.legend{display:flex;gap:20px;margin:10px 0 0;color:#8b949e;font-size:12px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
</style></head><body>
<header>
<h1>MausoleoBench &mdash; live progress</h1>
<div class=sub id=sub>loading&hellip;</div>
</header>
<div id=wrap>
<div class=legend>
<span><span class=dot style="background:#f2cc60"></span>new record (best so far)</span>
<span><span class=dot style="background:#484f58"></span>attempt</span>
<span><span class=dot style="background:#f2cc60;border-radius:0;width:16px;height:3px"></span>best-so-far frontier</span>
<span><span class=dot style="background:#a371f7;border-radius:0;width:16px;height:3px"></span>oracle ceiling</span>
</div>
<canvas id=c width=1100 height=460></canvas>
<h2 style="font-size:15px;margin:24px 0 0">Record-setting experiments</h2>
<table id=t><thead><tr><th>#</th><th>MausoleoBench</th><th>&Delta;</th><th>experiment</th><th>description</th></tr></thead><tbody></tbody></table>
</div>
<script>
const C=document.getElementById('c'),X=C.getContext('2d');
function draw(d){
  const W=C.width,H=C.height,padL=54,padR=20,padT=20,padB=42;
  X.clearRect(0,0,W,H);
  const A=d.attempts.slice().sort((a,b)=>a.n-b.n);
  if(!A.length)return;
  const refs=d.references||[];
  const allS=A.map(a=>a.score).concat(refs.map(r=>r.score));
  const ymax=Math.max(0.8,Math.ceil(Math.max(...allS)*10)/10+0.05), ymin=0;
  const n=A.length;
  const xat=i=>padL+(n<2?0.5:i/(n-1))*(W-padL-padR);
  const yat=v=>padT+(1-(v-ymin)/(ymax-ymin))*(H-padT-padB);
  X.strokeStyle='#21262d';X.fillStyle='#8b949e';X.font='11px sans-serif';X.lineWidth=1;
  for(let g=0;g<=ymax*10+1e-9;g++){const v=g/10;const y=yat(v);
    X.beginPath();X.moveTo(padL,y);X.lineTo(W-padR,y);X.stroke();
    X.fillText(v.toFixed(1),8,y+3);}
  refs.forEach(r=>{const y=yat(r.score);X.strokeStyle='#a371f7';X.setLineDash([6,5]);
    X.beginPath();X.moveTo(padL,y);X.lineTo(W-padR,y);X.stroke();X.setLineDash([]);
    X.fillStyle='#a371f7';X.fillText(r.name+' '+r.score.toFixed(3),W-padR-190,y-5);});
  let best=0,fr=[];
  A.forEach((a,i)=>{a._rec=a.score>best+1e-9;if(a._rec){best=a.score;fr.push([xat(i),yat(a.score)]);}});
  if(fr.length){X.strokeStyle='#f2cc60';X.lineWidth=2;X.beginPath();
    X.moveTo(fr[0][0],fr[0][1]);
    for(let i=1;i<fr.length;i++){X.lineTo(fr[i][0],fr[i-1][1]);X.lineTo(fr[i][0],fr[i][1]);}
    X.lineTo(xat(n-1),fr[fr.length-1][1]);X.stroke();}
  A.forEach((a,i)=>{const x=xat(i),y=yat(a.score);
    if(a._rec){X.fillStyle='#f2cc60';X.beginPath();X.arc(x,y,5.5,0,7);X.fill();}
    else{X.fillStyle='#484f58';X.beginPath();X.arc(x,y,3,0,7);X.fill();}});
  X.fillStyle='#8b949e';X.fillText('experiment attempt →',W-160,H-10);
}
function fmt(x){return (x>=0?'+':'')+x.toFixed(3);}
async function tick(){
  const d=await (await fetch('/data')).json();
  draw(d);
  const recs=d.attempts.filter(a=>a.score>0).sort((a,b)=>a.n-b.n);
  let best=0;const rows=[];let prev=0;
  recs.forEach(a=>{if(a.score>best+1e-9){rows.push([a,a.score-prev]);prev=a.score;best=a.score;}});
  document.querySelector('#t tbody').innerHTML=rows.slice().reverse().map(([a,dl])=>
    `<tr class=rec><td class=n>${a.n}</td><td class=score>${a.score.toFixed(4)}</td>`+
    `<td class=delta>${fmt(dl)}</td><td class=name>${a.config}</td><td>${a.description||''}</td></tr>`).join('');
  const b=d.attempts.reduce((m,a)=>Math.max(m,a.score),0);
  document.getElementById('sub').textContent=
    `${d.attempts.length} attempts · best ${b.toFixed(4)} · ${rows.length} records · auto-refresh 5s`;
}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a: tp.Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/data"):
            body = json.dumps(build_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main() -> None:
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print(f"progress dashboard on :{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
