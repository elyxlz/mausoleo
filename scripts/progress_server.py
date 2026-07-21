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


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MausoleoBench — live progress</title>
<style>
:root{
  color-scheme:dark;
  --page:#0b0e13;--card:#12161c;--edge:#222a34;--grid:#1e242d;--baseline:#39424d;
  --ink:#f2f5f7;--ink2:#a9b4bf;--ink3:#75808c;
  --gold:#c98500;--gold-bright:#e8b04a;--violet:#9085e9;--mute-dot:#5b6673;--green:#0ca30c;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1100px;margin:0 auto;padding:28px 24px 56px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:20px;flex-wrap:wrap}
h1{margin:0;font-size:21px;font-weight:650;letter-spacing:-.01em}
h1 .dim{color:var(--ink3);font-weight:500}
#statusline{margin:6px 0 0;font-size:13px;color:var(--ink2)}
#statusline b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
#statusline .sep{color:var(--ink3);margin:0 7px}
.live{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ink3);white-space:nowrap;padding-bottom:3px}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2.4s ease-out infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.45)}70%{box-shadow:0 0 0 7px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
.stale .pulse{background:#d03b3b;animation:none}
.stale #livetext{color:#d03b3b}
.card{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:18px 20px}
.card+.card{margin-top:18px}
.cardhead{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.cardtitle{font-size:11px;font-weight:650;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3)}
#expcount{font-weight:500;letter-spacing:0;text-transform:none;margin-left:6px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--ink2)}
.legend .sw{display:inline-block;vertical-align:-1px;margin-right:6px}
svg{display:block;width:100%;height:auto}
svg text{font-family:inherit}
.dot{transition:r .15s ease}
.hit{fill:transparent}
#tip{position:fixed;left:0;top:0;z-index:10;max-width:340px;pointer-events:none;background:#1a2029;border:1px solid #2c3540;border-radius:10px;padding:11px 13px;font-size:12.5px;line-height:1.5;box-shadow:0 10px 28px rgba(0,0,0,.45);opacity:0;transition:opacity .12s ease}
#tip.show{opacity:1}
.tiphead{display:flex;justify-content:space-between;gap:14px;align-items:baseline}
.tipn{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.tipscore{font-weight:700;font-variant-numeric:tabular-nums;font-size:14px}
.tipcfg{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#b9c6d4;margin-top:3px;word-break:break-all}
.tipbadge{display:inline-block;margin-top:6px;font-size:11px;font-weight:650;color:var(--gold-bright);background:rgba(201,133,0,.14);border:1px solid rgba(201,133,0,.35);border-radius:999px;padding:1px 8px}
.tipdesc{color:var(--ink2);margin-top:6px}
.seg{display:inline-flex;background:var(--page);border:1px solid var(--edge);border-radius:8px;padding:3px;gap:2px}
.seg button{border:0;background:transparent;color:var(--ink2);font:inherit;font-size:12.5px;padding:5px 12px;border-radius:6px;cursor:pointer;transition:background .15s,color .15s}
.seg button.on{background:#232a33;color:var(--ink)}
.seg button:not(.on):hover{color:var(--ink)}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:6px 12px;color:var(--ink3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
td{padding:8px 12px;border-top:1px solid var(--grid);vertical-align:top}
tbody tr{animation:rowin .2s ease both;transition:background .12s}
tbody tr:hover{background:rgba(255,255,255,.03)}
@keyframes rowin{from{opacity:0;transform:translateY(2px)}}
.cn{color:var(--ink3);font-variant-numeric:tabular-nums;white-space:nowrap}
.cs{font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:not(.rec) .cs{color:var(--ink2);font-weight:500}
.cd{color:var(--green);font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12px}
.cd .nil{color:var(--ink3)}
.cc{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#b9c6d4;word-break:break-all}
.cx{color:var(--ink2)}
.recdot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--gold);margin-right:7px;vertical-align:1px}
.empty{color:var(--ink3);text-align:center;padding:22px}
</style></head><body>
<main>
<header>
  <div>
    <h1>MausoleoBench <span class=dim>— live progress</span></h1>
    <p id=statusline>connecting…</p>
  </div>
  <div class=live><span class=pulse></span><span id=livetext>live · 5s</span></div>
</header>
<section class=card>
  <div class=cardhead>
    <span class=cardtitle>MausoleoBench score vs attempt</span>
    <div class=legend>
      <span><span class=sw style="width:9px;height:9px;border-radius:50%;background:#c98500"></span>record</span>
      <span><span class=sw style="width:7px;height:7px;border-radius:50%;background:#5b6673"></span>attempt</span>
      <span><span class=sw style="width:18px;height:2px;background:#c98500"></span>best-so-far frontier</span>
    </div>
  </div>
  <svg id=chart viewBox="0 0 1040 400" role=img aria-label="MausoleoBench score versus experiment attempt"></svg>
</section>
<section class=card>
  <div class=cardhead>
    <span class=cardtitle>Experiments<span id=expcount></span></span>
    <div class=seg role=group aria-label="filter experiments">
      <button type=button data-mode=records class=on aria-pressed=true>Record-setting</button>
      <button type=button data-mode=all aria-pressed=false>All</button>
    </div>
  </div>
  <div class=tablewrap>
  <table>
    <thead><tr><th>#</th><th>score</th><th>Δ record</th><th>config</th><th>description</th></tr></thead>
    <tbody id=tbody></tbody>
  </table>
  </div>
</section>
</main>
<div id=tip></div>
<script>
(()=>{
'use strict';
const chartEl=document.getElementById('chart');
const tipEl=document.getElementById('tip');
const statusEl=document.getElementById('statusline');
const bodyEl=document.getElementById('tbody');
const countEl=document.getElementById('expcount');
const segBtns=Array.from(document.querySelectorAll('.seg button'));
const view={filter:'records',raw:'',data:null,attempts:[],deltas:{}};
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f4=v=>v.toFixed(4);
const plus=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(4);
const W=1040,H=400,padL=48,padR=18,padT=20,padB=44;
const rd=v=>Math.round(v*10)/10;

function prepare(d){
  const A=d.attempts.slice().sort((a,b)=>a.n-b.n);
  const deltas={};
  let best=0;
  A.forEach(a=>{if(a.record){deltas[a.n]=a.score-best;best=a.score;}});
  view.attempts=A;view.deltas=deltas;view.data=d;
}

function renderStatus(){
  const A=view.attempts;
  const recs=A.filter(a=>a.record).length;
  statusEl.innerHTML='<b>'+A.length+'</b> attempts<span class=sep>·</span>best <b>'+f4(view.data.best||0)+'</b>'
    +'<span class=sep>·</span><b>'+recs+'</b> records';
}

function tipHtml(a){
  const d=view.deltas[a.n];
  return '<div class=tiphead><span class=tipn>attempt '+a.n+'</span><span class=tipscore>'+f4(a.score)+'</span></div>'
    +'<div class=tipcfg>'+esc(a.config)+'</div>'
    +(a.record?'<span class=tipbadge>record '+plus(d)+' vs prior</span>':'')
    +(a.description?'<div class=tipdesc>'+esc(a.description)+'</div>':'');
}

function placeTip(ev){
  const r=tipEl.getBoundingClientRect();
  let x=ev.clientX+16,y=ev.clientY+16;
  if(x+r.width>window.innerWidth-10)x=ev.clientX-r.width-16;
  if(y+r.height>window.innerHeight-10)y=ev.clientY-r.height-16;
  tipEl.style.transform='translate('+Math.max(6,x)+'px,'+Math.max(6,y)+'px)';
}

function bindHover(){
  const guide=chartEl.querySelector('#guide');
  chartEl.querySelectorAll('.hit').forEach(h=>{
    const i=+h.dataset.i,a=view.attempts[i];
    const dot=chartEl.querySelector('.dot[data-i="'+i+'"]');
    h.addEventListener('mouseenter',()=>{
      tipEl.innerHTML=tipHtml(a);
      tipEl.classList.add('show');
      guide.setAttribute('x1',h.getAttribute('cx'));
      guide.setAttribute('x2',h.getAttribute('cx'));
      guide.setAttribute('opacity','.6');
      dot.setAttribute('r',a.record?'7.5':'5');
    });
    h.addEventListener('mousemove',placeTip);
    h.addEventListener('mouseleave',()=>{
      tipEl.classList.remove('show');
      guide.setAttribute('opacity','0');
      dot.setAttribute('r',a.record?'5.5':'3.2');
    });
  });
}

function renderChart(){
  const A=view.attempts;
  if(!A.length){chartEl.innerHTML='<text x="'+(W/2)+'" y="'+(H/2)+'" text-anchor="middle" fill="#75808c" font-size="13">no attempts yet</text>';return;}
  const minN=A[0].n,maxN=A[A.length-1].n;
  const hi=Math.max(...A.map(a=>a.score));
  const yMax=Math.max(.8,Math.ceil((hi+.04)*10)/10);
  const xAt=v=>maxN===minN?(padL+W-padR)/2:padL+(v-minN)/(maxN-minN)*(W-padL-padR);
  const yAt=v=>padT+(1-v/yMax)*(H-padT-padB);
  const P=[];
  P.push('<defs><linearGradient id="gf" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c98500" stop-opacity=".10"/><stop offset="1" stop-color="#c98500" stop-opacity="0"/></linearGradient></defs>');
  for(let g=0;g<=Math.round(yMax*10);g++){
    const y=rd(yAt(g/10));
    P.push('<line x1="'+padL+'" y1="'+y+'" x2="'+(W-padR)+'" y2="'+y+'" stroke="'+(g===0?'#39424d':'#1e242d')+'"/>');
    P.push('<text x="'+(padL-9)+'" y="'+(y+3.5)+'" text-anchor="end" fill="#75808c" font-size="11">'+(g/10).toFixed(1)+'</text>');
  }
  const stepN=Math.max(1,Math.ceil((maxN-minN)/14));
  for(let v=minN;v<=maxN;v+=stepN){
    P.push('<text x="'+rd(xAt(v))+'" y="'+(H-padB+18)+'" text-anchor="middle" fill="#75808c" font-size="11">'+v+'</text>');
  }
  P.push('<text x="'+rd((padL+W-padR)/2)+'" y="'+(H-8)+'" text-anchor="middle" fill="#75808c" font-size="11" letter-spacing=".06em">experiment attempt</text>');
  const recs=A.filter(a=>a.record);
  if(recs.length){
    let p='M'+rd(xAt(recs[0].n))+' '+rd(yAt(recs[0].score));
    for(let i=1;i<recs.length;i++)p+=' H'+rd(xAt(recs[i].n))+' V'+rd(yAt(recs[i].score));
    p+=' H'+rd(xAt(maxN));
    P.push('<path d="'+p+' V'+rd(yAt(0))+' H'+rd(xAt(recs[0].n))+' Z" fill="url(#gf)"/>');
    P.push('<path d="'+p+'" fill="none" stroke="#c98500" stroke-width="2" stroke-linejoin="round"/>');
  }
  P.push('<line id="guide" x1="0" y1="'+padT+'" x2="0" y2="'+(H-padB)+'" stroke="#3a4450" opacity="0"/>');
  const lastRec=recs.length?recs[recs.length-1]:null;
  A.forEach((a,i)=>{
    const x=rd(xAt(a.n)),y=rd(yAt(a.score));
    if(lastRec&&a.n===lastRec.n)P.push('<circle cx="'+x+'" cy="'+y+'" r="11" fill="#c98500" opacity=".14"/>');
    if(a.record)P.push('<circle class="dot rec" data-i="'+i+'" cx="'+x+'" cy="'+y+'" r="5.5" fill="#c98500" stroke="#12161c" stroke-width="2"/>');
    else P.push('<circle class="dot" data-i="'+i+'" cx="'+x+'" cy="'+y+'" r="3.2" fill="#5b6673"/>');
  });
  A.forEach((a,i)=>{
    P.push('<circle class="hit" data-i="'+i+'" cx="'+rd(xAt(a.n))+'" cy="'+rd(yAt(a.score))+'" r="14"/>');
  });
  chartEl.innerHTML=P.join('');
  bindHover();
}

function renderTable(){
  const rows=view.attempts.slice().sort((a,b)=>b.n-a.n).filter(a=>view.filter==='all'||a.record);
  countEl.textContent=' · '+rows.length+' of '+view.attempts.length;
  if(!rows.length){bodyEl.innerHTML='<tr><td colspan=5 class=empty>no experiments yet</td></tr>';return;}
  bodyEl.innerHTML=rows.map(a=>{
    const d=view.deltas[a.n];
    return '<tr'+(a.record?' class=rec':'')+'>'
      +'<td class=cn>'+(a.record?'<span class=recdot></span>':'')+a.n+'</td>'
      +'<td class=cs>'+f4(a.score)+'</td>'
      +'<td class=cd>'+(a.record?plus(d):'<span class=nil>—</span>')+'</td>'
      +'<td class=cc>'+esc(a.config)+'</td>'
      +'<td class=cx>'+esc(a.description||'')+'</td></tr>';
  }).join('');
}

segBtns.forEach(b=>b.addEventListener('click',()=>{
  if(view.filter===b.dataset.mode)return;
  view.filter=b.dataset.mode;
  segBtns.forEach(x=>{
    const on=x.dataset.mode===view.filter;
    x.classList.toggle('on',on);
    x.setAttribute('aria-pressed',String(on));
  });
  renderTable();
}));

async function refresh(){
  try{
    const res=await fetch('/data',{cache:'no-store'});
    const txt=await res.text();
    document.body.classList.remove('stale');
    if(txt===view.raw)return;
    view.raw=txt;
    prepare(JSON.parse(txt));
    renderStatus();
    renderChart();
    renderTable();
  }catch(_e){
    document.body.classList.add('stale');
  }
}
refresh();
setInterval(refresh,5000);
})();
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
