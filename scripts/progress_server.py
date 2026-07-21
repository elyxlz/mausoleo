from __future__ import annotations

import http.server
import json
import pathlib as pl
import re
import shutil
import socketserver
import typing as tp
import urllib.parse as up

LOG = pl.Path("eval/autoresearch/mausoleobench_log.jsonl")
PORT = 8078


def load_attempts() -> list[dict[str, tp.Any]]:
    if not LOG.exists():
        return []
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["n"])


BUDGET_CAP = 13.9


def build_payload() -> dict[str, tp.Any]:
    rows = load_attempts()
    attempts = [r for r in rows if not r.get("reference")]
    references = [r for r in rows if r.get("reference")]
    best = 0.0
    for r in sorted(attempts, key=lambda r: r["n"]):
        r["budget_ok"] = bool(r.get("budget_ok"))
        r["record"] = r["budget_ok"] and r["score"] > best + 1e-9
        if r["record"]:
            best = r["score"]
    return {"attempts": attempts, "references": references, "best": best, "budget_cap": BUDGET_CAP}


VIEWER_DATES = ("1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15")
PREDICTIONS_DIR = pl.Path("eval/predictions")
IMAGES_DIR = pl.Path("eval/ground_truth")
CONFIG_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def find_prediction_path(config: str, date: str) -> pl.Path | None:
    if not CONFIG_PATTERN.match(config) or date not in VIEWER_DATES:
        return None
    path = PREDICTIONS_DIR / f"{config}_{date}.json"
    return path if path.is_file() else None


def count_pages(date: str) -> int:
    issue_dir = IMAGES_DIR / date
    if not issue_dir.is_dir():
        return 0
    return sum(1 for p in issue_dir.glob("*.jpeg") if p.stem.isdigit())


def find_attempt_n(config: str) -> int | None:
    for row in load_attempts():
        if row.get("config") == config and not row.get("reference"):
            return row.get("n")
    return None


def build_viewer_meta(config: str) -> dict[str, tp.Any]:
    dates = [
        {
            "date": d,
            "has_prediction": find_prediction_path(config, d) is not None,
            "page_count": count_pages(d),
        }
        for d in VIEWER_DATES
    ]
    return {"config": config, "n": find_attempt_n(config), "dates": dates}


def find_image_path(date: str, page: str) -> pl.Path | None:
    if date not in VIEWER_DATES or not page.isdigit():
        return None
    path = IMAGES_DIR / date / f"{int(page)}.jpeg"
    return path if path.is_file() else None


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MausoleoBench — live progress</title>
<style>
:root{
  color-scheme:dark;
  --page:#0b0e13;--card:#12161c;--edge:#222a34;--grid:#1e242d;--baseline:#39424d;
  --ink:#f2f5f7;--ink2:#a9b4bf;--ink3:#75808c;
  --gold:#c98500;--gold-bright:#e8b04a;--violet:#9085e9;--mute-dot:#5b6673;--green:#0ca30c;
  --dq:#a06a6a;--dq-bright:#c98a8a;
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
.hit{fill:transparent;cursor:pointer}
#tip{position:fixed;left:0;top:0;z-index:10;max-width:340px;pointer-events:none;background:#1a2029;border:1px solid #2c3540;border-radius:10px;padding:11px 13px;font-size:12.5px;line-height:1.5;box-shadow:0 10px 28px rgba(0,0,0,.45);opacity:0;transition:opacity .12s ease}
#tip.show{opacity:1}
.tiphead{display:flex;justify-content:space-between;gap:14px;align-items:baseline}
.tipexp{font-weight:700;font-size:13px;color:var(--ink)}
.tipscore{font-weight:700;font-variant-numeric:tabular-nums;font-size:14px}
.tipcfg{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#b9c6d4;margin-top:3px;word-break:break-all}
.tipcost{margin-top:5px;font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink2)}
.tipcost.bad{color:var(--dq-bright);font-weight:600}
.tipbadge{display:inline-block;margin-top:6px;font-size:11px;font-weight:650;color:var(--gold-bright);background:rgba(201,133,0,.14);border:1px solid rgba(201,133,0,.35);border-radius:999px;padding:1px 8px}
.tipdesc{color:var(--ink2);margin-top:6px}
.tipopen{margin-top:7px;font-size:11px;color:var(--ink3)}
.seg{display:inline-flex;background:var(--page);border:1px solid var(--edge);border-radius:8px;padding:3px;gap:2px}
.seg button{border:0;background:transparent;color:var(--ink2);font:inherit;font-size:12.5px;padding:5px 12px;border-radius:6px;cursor:pointer;transition:background .15s,color .15s}
.seg button.on{background:#232a33;color:var(--ink)}
.seg button:not(.on):hover{color:var(--ink)}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:6px 12px;color:var(--ink3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
td{padding:8px 12px;border-top:1px solid var(--grid);vertical-align:top}
tbody tr{animation:rowin .2s ease both;transition:background .12s;cursor:pointer}
tbody tr:hover{background:rgba(255,255,255,.03)}
@keyframes rowin{from{opacity:0;transform:translateY(2px)}}
.cn{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:650;color:var(--ink)}
tbody tr.dq .cn{color:var(--ink2)}
.cs{font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:not(.rec) .cs{color:var(--ink2);font-weight:500}
.cd{color:var(--green);font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12px}
.cd .nil{color:var(--ink3)}
.cg{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12px;color:var(--ink2)}
.cg .dqtag{color:var(--dq-bright);font-weight:600}
.cc{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#b9c6d4;word-break:break-all;text-decoration:none;border-bottom:1px dotted #3a4450}
a.cc:hover{color:var(--ink);border-bottom-color:var(--gold)}
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
      <span><span class=sw style="width:7px;height:7px;border-radius:50%;background:#5b6673"></span>in-budget attempt</span>
      <span><span class=sw style="width:8px;height:8px;border-radius:50%;border:1.5px solid #a06a6a;background:transparent"></span>over budget (disqualified)</span>
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
    <thead><tr><th>exp</th><th>score</th><th>Δ record</th><th>GPU-s/page</th><th>pipeline</th><th>description</th></tr></thead>
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
const fmtCost=v=>v==null?null:(Number.isInteger(v)?String(v):(Math.round(v*10)/10).toFixed(1))+' GPU-s/page';
const viewerUrl=a=>'/viewer?config='+encodeURIComponent(a.config);
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
  const dq=A.filter(a=>!a.budget_ok).length;
  statusEl.innerHTML='<b>'+A.length+'</b> attempts<span class=sep>·</span>best in-budget <b>'+f4(view.data.best||0)+'</b>'
    +'<span class=sep>·</span><b>'+recs+'</b> records'
    +'<span class=sep>·</span><b>'+dq+'</b> over budget';
}

function costHtml(a){
  const cost=fmtCost(a.gpu_s_per_page);
  if(!a.budget_ok)return '<div class="tipcost bad">'+(cost?cost+' · ':'')+'over budget · disqualified</div>';
  return cost?'<div class=tipcost>'+cost+' · within budget</div>':'';
}

function tipHtml(a){
  const d=view.deltas[a.n];
  return '<div class=tiphead><span class=tipexp>Exp '+a.n+'</span><span class=tipscore>'+f4(a.score)+'</span></div>'
    +'<div class=tipcfg>'+esc(a.config)+'</div>'
    +costHtml(a)
    +(a.record?'<span class=tipbadge>record '+plus(d)+' vs prior</span>':'')
    +(a.description?'<div class=tipdesc>'+esc(a.description)+'</div>':'')
    +'<div class=tipopen>click to open prediction viewer ↗</div>';
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
    const baseR=dot.dataset.r;
    h.addEventListener('mouseenter',()=>{
      tipEl.innerHTML=tipHtml(a);
      tipEl.classList.add('show');
      guide.setAttribute('x1',h.getAttribute('cx'));
      guide.setAttribute('x2',h.getAttribute('cx'));
      guide.setAttribute('opacity','.6');
      dot.setAttribute('r',String(+baseR+2));
    });
    h.addEventListener('mousemove',placeTip);
    h.addEventListener('mouseleave',()=>{
      tipEl.classList.remove('show');
      guide.setAttribute('opacity','0');
      dot.setAttribute('r',baseR);
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
    if(a.record)P.push('<circle class="dot rec" data-i="'+i+'" data-r="5.5" cx="'+x+'" cy="'+y+'" r="5.5" fill="#c98500" stroke="#12161c" stroke-width="2"/>');
    else if(a.budget_ok)P.push('<circle class="dot" data-i="'+i+'" data-r="3.2" cx="'+x+'" cy="'+y+'" r="3.2" fill="#5b6673"/>');
    else P.push('<circle class="dot dq" data-i="'+i+'" data-r="3.6" cx="'+x+'" cy="'+y+'" r="3.6" fill="none" stroke="#a06a6a" stroke-width="1.6" opacity=".85"/>');
  });
  A.forEach((a,i)=>{
    P.push('<a class=ptlink href="'+viewerUrl(a)+'" target=_blank rel=noopener>'
      +'<circle class="hit" data-i="'+i+'" cx="'+rd(xAt(a.n))+'" cy="'+rd(yAt(a.score))+'" r="14"/></a>');
  });
  chartEl.innerHTML=P.join('');
  bindHover();
}

function costCell(a){
  const cost=fmtCost(a.gpu_s_per_page);
  if(!a.budget_ok)return '<td class=cg>'+(cost?cost:'')+'<span class=dqtag>'+(cost?' · ':'')+'over budget · disqualified</span></td>';
  return '<td class=cg>'+(cost?cost:'<span class=nil>—</span>')+'</td>';
}

function renderTable(){
  const rows=view.attempts.slice().sort((a,b)=>b.n-a.n).filter(a=>view.filter==='all'||a.record);
  countEl.textContent=' · '+rows.length+' of '+view.attempts.length;
  if(!rows.length){bodyEl.innerHTML='<tr><td colspan=6 class=empty>no experiments yet</td></tr>';return;}
  bodyEl.innerHTML=rows.map(a=>{
    const d=view.deltas[a.n];
    const cls=(a.record?'rec ':'')+(a.budget_ok?'':'dq');
    return '<tr'+(cls?' class="'+cls+'"':'')+' data-url="'+viewerUrl(a)+'">'
      +'<td class=cn>'+(a.record?'<span class=recdot></span>':'')+'Exp '+a.n+'</td>'
      +'<td class=cs>'+f4(a.score)+'</td>'
      +'<td class=cd>'+(a.record?plus(d):'<span class=nil>—</span>')+'</td>'
      +costCell(a)
      +'<td><a class=cc href="'+viewerUrl(a)+'" target=_blank rel=noopener>'+esc(a.config)+'</a></td>'
      +'<td class=cx>'+esc(a.description||'')+'</td></tr>';
  }).join('');
}

bodyEl.addEventListener('click',ev=>{
  if(ev.target.closest('a'))return;
  const row=ev.target.closest('tr[data-url]');
  if(row)window.open(row.dataset.url,'_blank','noopener');
});

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


VIEWER_PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>prediction viewer</title>
<style>
:root{
  color-scheme:dark;
  --page:#0b0e13;--card:#12161c;--edge:#222a34;--grid:#1e242d;
  --ink:#f2f5f7;--ink2:#a9b4bf;--ink3:#75808c;
  --gold:#c98500;--gold-bright:#e8b04a;--accent:#4da3ff;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:20;background:var(--card);border-bottom:1px solid var(--edge);padding:10px 18px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header a.back{color:var(--ink3);text-decoration:none;font-size:13px;white-space:nowrap}
header a.back:hover{color:var(--ink)}
#vtitle{font-size:16px;font-weight:650;letter-spacing:-.01em;white-space:nowrap}
#vcfg{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#b9c6d4;word-break:break-all}
.ro{font-size:10px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);border:1px solid var(--edge);border-radius:999px;padding:2px 8px;white-space:nowrap}
.seg{display:inline-flex;background:var(--page);border:1px solid var(--edge);border-radius:8px;padding:3px;gap:2px;flex-wrap:wrap}
.seg button{border:0;background:transparent;color:var(--ink2);font:inherit;font-size:12px;padding:4px 9px;border-radius:6px;cursor:pointer;font-variant-numeric:tabular-nums;transition:background .15s,color .15s}
.seg button.on{background:#232a33;color:var(--ink)}
.seg button:not(.on):hover{color:var(--ink)}
.seg button.miss{opacity:.45}
.pager{display:flex;gap:6px;align-items:center;font-size:12.5px;color:var(--ink2);white-space:nowrap}
.pager button,.pager select{border:1px solid var(--edge);background:var(--page);color:var(--ink2);font:inherit;font-size:12.5px;border-radius:6px;padding:3px 8px;cursor:pointer}
.pager button:hover,.pager select:hover{color:var(--ink)}
#unitcount{margin-left:auto;font-size:12px;color:var(--ink3);white-space:nowrap}
main{display:flex;align-items:flex-start}
#imgpane{width:55%;position:sticky;top:49px;height:calc(100vh - 49px);background:#161a20}
#viewport{width:100%;height:100%;overflow:auto;position:relative;cursor:crosshair}
#viewport.panning{cursor:grabbing}
#viewport.canpan{cursor:grab}
#canvas{position:relative;width:100%}
#pageimg{width:100%;display:block;user-select:none;-webkit-user-drag:none}
#marquee{position:absolute;border:2px solid var(--accent);background:rgba(77,163,255,.18);display:none;pointer-events:none;z-index:5}
#hud{position:absolute;left:10px;bottom:10px;z-index:10;display:flex;gap:5px;align-items:center;background:rgba(11,14,19,.78);color:var(--ink);padding:5px 8px;border:1px solid var(--edge);border-radius:8px;font:12px ui-monospace,Menlo,Consolas,monospace}
#hud button{font:12px ui-monospace,Menlo,Consolas,monospace;cursor:pointer;border:0;border-radius:5px;padding:2px 8px;background:#232a33;color:var(--ink)}
#hud button:hover{background:#2e3742}
#hint{position:absolute;right:10px;top:10px;z-index:10;background:rgba(11,14,19,.72);color:var(--ink2);padding:6px 9px;border:1px solid var(--edge);border-radius:8px;font-size:11px;max-width:17em;line-height:1.4}
#units{width:45%;padding:14px 18px 80px}
.unit{background:var(--card);border:1px solid var(--edge);border-radius:10px;margin-bottom:12px;padding:12px 15px}
.unit.multi{border-left:3px solid #9085e9}
.ubar{display:flex;gap:10px;align-items:baseline;font-size:11px;color:var(--ink3);margin-bottom:6px}
.ubar .uid{font-variant-numeric:tabular-nums}
.hl{margin:0 0 7px;font:650 15px/1.35 Georgia,serif;color:var(--ink)}
.hl.none{color:var(--ink3);font-weight:400;font-style:italic;font-size:13px}
.para{margin:0 0 9px;font:13.5px/1.55 Georgia,serif;color:#d7dde3;white-space:pre-wrap}
.para:last-child{margin-bottom:0}
.nopred{background:var(--card);border:1px dashed var(--edge);border-radius:10px;padding:34px 20px;text-align:center;color:var(--ink3)}
.nopred b{display:block;color:var(--ink2);font-size:15px;margin-bottom:5px}
</style></head><body>
<header>
  <a class=back href="/">&#8592; dashboard</a>
  <span id=vtitle>&#8230;</span>
  <span id=vcfg></span>
  <span class=ro>read-only</span>
  <div class=seg id=dateseg role=group aria-label="issue date"></div>
  <div class=pager>
    <button type=button id=prevpg>&#8592;</button>
    <span>page <select id=pagesel></select> / <span id=pagecount>?</span></span>
    <button type=button id=nextpg>&#8594;</button>
  </div>
  <span id=unitcount></span>
</header>
<main>
  <div id=imgpane>
    <div id=viewport>
      <div id=canvas><img id=pageimg alt="newspaper page scan"><div id=marquee></div></div>
    </div>
    <div id=hud>
      <button type=button id=zoomout>&#8722;</button>
      <span id=zoomlbl>100%</span>
      <button type=button id=zoomin>+</button>
      <button type=button id=backbtn title="step back out (Esc)">&#8617; back</button>
      <button type=button id=resetbtn title="full page">reset</button>
      <span id=depthlbl></span>
    </div>
    <div id=hint>Drag a box to zoom in &#183; click (or Esc) to zoom back out &#183; Space+drag or scroll to pan &#183; &#8984;/Ctrl+scroll to fine-zoom</div>
  </div>
  <div id=units></div>
</main>
<script>
(()=>{
'use strict';
const config=new URLSearchParams(location.search).get('config')||'';
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const el=id=>document.getElementById(id);
const viewport=el('viewport'),canvas=el('canvas'),img=el('pageimg'),marquee=el('marquee');
const state={meta:null,doc:null,date:null,page:1};
const MAXZOOM=8;
let stack=[{zoom:1,cx:.5,cy:0}];
let spaceDown=false,drag=null;

function paneW(){return viewport.clientWidth}
function curView(){return stack[stack.length-1]}

function applyView(){
  const f=curView();
  const w=Math.max(paneW(),paneW()*f.zoom);
  canvas.style.width=w+'px';
  const rh=canvas.clientHeight||1;
  const maxL=Math.max(0,w-viewport.clientWidth);
  const maxT=Math.max(0,rh-viewport.clientHeight);
  viewport.scrollLeft=Math.min(maxL,Math.max(0,f.cx*w-viewport.clientWidth/2));
  viewport.scrollTop=Math.min(maxT,Math.max(0,f.cy*rh-viewport.clientHeight/2));
  el('zoomlbl').textContent=Math.round(f.zoom*100)+'%';
  el('depthlbl').textContent=stack.length>1?'\\u25a2\\u00d7'+(stack.length-1):'';
  viewport.classList.toggle('canpan',f.zoom>1.001);
}

function centerFractions(){
  const w=canvas.clientWidth||1,h=canvas.clientHeight||1;
  return {cx:(viewport.scrollLeft+viewport.clientWidth/2)/w,cy:(viewport.scrollTop+viewport.clientHeight/2)/h};
}

function resetView(){stack=[{zoom:1,cx:.5,cy:0}];applyView();}
function popView(){if(stack.length>1){stack.pop();applyView();}else resetView();}

function pushBox(fx0,fy0,fx1,fy1){
  const bw=Math.max(.01,fx1-fx0),bh=Math.max(.01,fy1-fy0);
  const aspect=(canvas.clientHeight||1)/(canvas.clientWidth||1);
  const zW=1/bw;
  const zH=viewport.clientHeight/(bh*paneW()*aspect);
  const zoom=Math.max(1,Math.min(MAXZOOM,Math.min(zW,zH)));
  stack.push({zoom,cx:(fx0+fx1)/2,cy:(fy0+fy1)/2});
  applyView();
}

function zoomStep(factor){
  const f=curView();
  const nz=Math.max(1,Math.min(MAXZOOM,f.zoom*factor));
  const c=centerFractions();
  if(Math.abs(nz-f.zoom)<1e-3)return;
  if(f.zoom===1&&nz>1)stack.push({zoom:nz,cx:c.cx,cy:c.cy});
  else{f.zoom=nz;f.cx=c.cx;f.cy=c.cy;}
  applyView();
}

function zoomAt(factor,clientX,clientY){
  const f=curView();
  const nz=Math.max(1,Math.min(MAXZOOM,f.zoom*factor));
  if(Math.abs(nz-f.zoom)<1e-3)return;
  const r=viewport.getBoundingClientRect();
  const w=canvas.clientWidth||1,h=canvas.clientHeight||1;
  const fracX=(viewport.scrollLeft+(clientX-r.left))/w;
  const fracY=(viewport.scrollTop+(clientY-r.top))/h;
  const target=(f.zoom===1&&nz>1)?{zoom:nz,cx:fracX,cy:fracY}:f;
  target.zoom=nz;
  if(f.zoom===1&&nz>1)stack.push(target);
  const w2=paneW()*nz,h2=w2*((canvas.clientHeight||1)/(canvas.clientWidth||1));
  const sl=fracX*w2-(clientX-r.left);
  const st=fracY*h2-(clientY-r.top);
  target.cx=(sl+viewport.clientWidth/2)/w2;
  target.cy=(st+viewport.clientHeight/2)/h2;
  applyView();
}

window.addEventListener('keydown',e=>{
  if(e.code==='Space'&&e.target.tagName!=='SELECT'){spaceDown=true;viewport.classList.add('canpan');e.preventDefault();}
  if(e.key==='Escape')popView();
});
window.addEventListener('keyup',e=>{if(e.code==='Space'){spaceDown=false;viewport.classList.remove('canpan');}});
viewport.addEventListener('contextmenu',e=>{e.preventDefault();popView();});

viewport.addEventListener('mousedown',e=>{
  if(e.button!==0)return;
  if(spaceDown||curView().zoom>1.001&&e.shiftKey){
    drag={mode:'pan',x:e.clientX,y:e.clientY,sl:viewport.scrollLeft,st:viewport.scrollTop};
    viewport.classList.add('panning');
  }else{
    drag={mode:'box',x0:e.clientX,y0:e.clientY};
  }
  e.preventDefault();
});

window.addEventListener('mousemove',e=>{
  if(!drag)return;
  if(drag.mode==='pan'){
    viewport.scrollLeft=drag.sl-(e.clientX-drag.x);
    viewport.scrollTop=drag.st-(e.clientY-drag.y);
    return;
  }
  const cr=canvas.getBoundingClientRect();
  const x0=Math.min(drag.x0,e.clientX),y0=Math.min(drag.y0,e.clientY);
  const x1=Math.max(drag.x0,e.clientX),y1=Math.max(drag.y0,e.clientY);
  marquee.style.display='block';
  marquee.style.left=(x0-cr.left)+'px';
  marquee.style.top=(y0-cr.top)+'px';
  marquee.style.width=(x1-x0)+'px';
  marquee.style.height=(y1-y0)+'px';
});

window.addEventListener('mouseup',e=>{
  if(!drag)return;
  if(drag.mode==='pan'){
    const c=centerFractions();curView().cx=c.cx;curView().cy=c.cy;
    viewport.classList.remove('panning');drag=null;return;
  }
  marquee.style.display='none';
  const cr=canvas.getBoundingClientRect();
  const w=canvas.clientWidth||1,h=canvas.clientHeight||1;
  const dx=Math.abs(e.clientX-drag.x0),dy=Math.abs(e.clientY-drag.y0);
  if(dx>8&&dy>8){
    const fx0=(Math.min(drag.x0,e.clientX)-cr.left)/w;
    const fy0=(Math.min(drag.y0,e.clientY)-cr.top)/h;
    const fx1=(Math.max(drag.x0,e.clientX)-cr.left)/w;
    const fy1=(Math.max(drag.y0,e.clientY)-cr.top)/h;
    pushBox(Math.max(0,fx0),Math.max(0,fy0),Math.min(1,fx1),Math.min(1,fy1));
  }else if(dx<6&&dy<6&&stack.length>1){
    popView();
  }
  drag=null;
});

viewport.addEventListener('wheel',e=>{
  if(e.ctrlKey||e.metaKey){e.preventDefault();zoomAt(e.deltaY<0?1.18:1/1.18,e.clientX,e.clientY);}
},{passive:false});
window.addEventListener('resize',applyView);

el('zoomout').addEventListener('click',()=>zoomStep(1/1.4));
el('zoomin').addEventListener('click',()=>zoomStep(1.4));
el('backbtn').addEventListener('click',popView);
el('resetbtn').addEventListener('click',resetView);

function dateInfo(d){return state.meta.dates.find(x=>x.date===d);}

function renderDateSeg(){
  const seg=el('dateseg');
  seg.innerHTML=state.meta.dates.map(d=>
    '<button type=button data-date="'+d.date+'" class="'+(d.date===state.date?'on':'')+(d.has_prediction?'':' miss')+'"'
    +(d.has_prediction?'':' title="no prediction"')+'>'+d.date+'</button>').join('');
  seg.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectDate(b.dataset.date)));
}

function renderPager(){
  const info=dateInfo(state.date);
  const sel=el('pagesel');
  sel.innerHTML='';
  for(let p=1;p<=info.page_count;p++)sel.append(new Option(String(p),String(p)));
  sel.value=String(state.page);
  el('pagecount').textContent=String(info.page_count);
}

function articleHtml(a){
  const span=a.page_span||[1];
  const multi=span.length>1;
  return '<article class="unit'+(multi?' multi':'')+'">'
    +(multi?'<div class=ubar><span class=uid>pages '+span.join(', ')+'</span></div>':'')
    +(a.headline?'<h3 class=hl>'+esc(a.headline)+'</h3>':'<h3 class="hl none">(no headline)</h3>')
    +(a.paragraphs||[]).map(p=>'<p class=para>'+esc(p.text)+'</p>').join('')
    +'</article>';
}

function renderArticles(){
  const box=el('units');
  if(!state.doc){
    box.innerHTML='<div class=nopred><b>no prediction</b>this experiment has no prediction file for '+esc(state.date)+'</div>';
    el('unitcount').textContent='';
    return;
  }
  const onPage=state.doc.articles.filter(a=>(a.page_span||[1]).includes(state.page));
  el('unitcount').textContent=onPage.length+' on page \\u00b7 '+state.doc.articles.length+' total';
  box.innerHTML=onPage.length
    ?onPage.map(articleHtml).join('')
    :'<div class=nopred>no predicted articles on this page</div>';
}

function showPage(){
  renderPager();
  img.onload=resetView;
  img.src='/viewer/img?date='+encodeURIComponent(state.date)+'&page='+state.page;
  renderArticles();
  window.scrollTo(0,0);
}

function shiftPage(d){
  const info=dateInfo(state.date);
  state.page=Math.min(info.page_count,Math.max(1,state.page+d));
  showPage();
}

el('prevpg').addEventListener('click',()=>shiftPage(-1));
el('nextpg').addEventListener('click',()=>shiftPage(1));
el('pagesel').addEventListener('change',()=>{state.page=+el('pagesel').value;showPage();});

async function selectDate(d){
  state.date=d;
  state.page=1;
  state.doc=null;
  renderDateSeg();
  if(dateInfo(d).has_prediction){
    const res=await fetch('/viewer/prediction?config='+encodeURIComponent(config)+'&date='+encodeURIComponent(d));
    if(res.ok)state.doc=await res.json();
  }
  showPage();
}

async function init(){
  const res=await fetch('/viewer/meta?config='+encodeURIComponent(config));
  if(!res.ok){
    el('vtitle').textContent='unknown experiment';
    el('units').innerHTML='<div class=nopred><b>unknown experiment</b>'+esc(config||'(no config given)')+'</div>';
    return;
  }
  state.meta=await res.json();
  const label=state.meta.n!=null?'Exp '+state.meta.n:config;
  el('vtitle').textContent=label;
  el('vcfg').textContent=config;
  document.title=label+' \\u2014 prediction viewer';
  const first=state.meta.dates.find(d=>d.has_prediction)||state.meta.dates[0];
  selectDate(first.date);
}

init();
})();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: tp.Any) -> None:
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: pl.Path, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _serve_viewer_meta(self, query: dict[str, str]) -> None:
        config = query.get("config", "")
        if not CONFIG_PATTERN.match(config):
            self._send(400, b'{"error":"bad config"}', "application/json")
            return
        self._send(200, json.dumps(build_viewer_meta(config)).encode(), "application/json")

    def _serve_viewer_prediction(self, query: dict[str, str]) -> None:
        path = find_prediction_path(query.get("config", ""), query.get("date", ""))
        if path is None:
            self._send(404, b'{"error":"no prediction"}', "application/json")
            return
        self._send_file(path, "application/json")

    def _serve_viewer_img(self, query: dict[str, str]) -> None:
        path = find_image_path(query.get("date", ""), query.get("page", ""))
        if path is None:
            self._send(404, b"not found", "text/plain")
            return
        self._send_file(path, "image/jpeg")

    def do_GET(self) -> None:
        if self.path.startswith("/data"):
            body = json.dumps(build_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        url = up.urlsplit(self.path)
        query = {k: v[0] for k, v in up.parse_qs(url.query).items()}
        if url.path == "/viewer":
            self._send(200, VIEWER_PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/viewer/meta":
            self._serve_viewer_meta(query)
        elif url.path == "/viewer/prediction":
            self._serve_viewer_prediction(query)
        elif url.path == "/viewer/img":
            self._serve_viewer_img(query)
        elif url.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main() -> None:
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print(f"progress dashboard on :{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
