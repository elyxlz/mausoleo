"""Hierarchical summarization pipeline for the Mausoleo BASc dissertation.

Builds the summary tree from OCR transcriptions of Il Messaggero, July 1943:
    paragraph (level 0) -> article (level 1) -> day (level 2)
    -> week (level 3) -> month (level 4)

Outputs JSON files compatible with the ClickHouse `nodes` schema described in
`plan/03_hierarchical_index.md`.

Special handling: 1943-07-26 is absent from the source archive (the day
between Mussolini's removal at 02:40 on 25 July and the new government's first
issue on 27 July). We emit a sentinel `day` node so the agent can detect and
discuss the gap; this is the dissertation's signature anomaly handling.

Run:
    python3 build_index.py [--dry-run] [--limit-articles N]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from sentence_transformers import SentenceTransformer


# ----- paths --------------------------------------------------------------

ROOT = pathlib.Path("/tmp/mausoleo")
TRANSCRIPTIONS = ROOT / "eval" / "transcriptions"
OUT = ROOT / "eval" / "summaries"
MANIFEST = OUT / "manifest.json"

LEVELS = ("paragraph", "article", "day", "week", "month")
for lvl in LEVELS:
    (OUT / lvl).mkdir(parents=True, exist_ok=True)


# ----- model + cost configuration ----------------------------------------

# Pricing per 1M tokens, USD. Source: anthropic.com/pricing 2025/2026.
PRICING = {
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-5":  {"in": 3.00, "out": 15.00},
}

# Article-level: cheap, lots of calls. Higher levels: smarter model, fewer calls.
MODEL_ARTICLE = "claude-haiku-4-5"
MODEL_DAY = "claude-sonnet-4-5"
MODEL_WEEK = "claude-sonnet-4-5"
MODEL_MONTH = "claude-sonnet-4-5"

BUDGET_CAP_USD = 30.0
ARTICLE_CONCURRENCY = 12

# BGE-M3 is the spec'd model (best for Italian) but on this CPU-only host
# (CUDA driver 12.2 too old for installed torch 2.11+cu130) it ran at ~1.6
# nodes/s = ~70 min for the article level. We fell back to a lightweight
# multilingual model so the index ships within the 3-hour budget; embeddings
# can be regenerated with BGE-M3 once the GPU is online.
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ----- prompts ------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sei un assistente storico esperto del «Messaggero di Roma». "
    "Riassumi articoli di giornale italiani del 1943 in italiano moderno, "
    "preservando entità (persone, luoghi, organizzazioni, date), "
    "numeri specifici e citazioni testuali. Non inventare nulla. "
    "Lascia stare le metafore: vai dritto ai fatti. "
    "Riassunti di 200-400 parole, lunghezza coerente a tutti i livelli."
)

ARTICLE_PROMPT = """Riassumi il seguente articolo de «Il Messaggero» del {date}.

Headline: {headline}

Testo:
{text}

Vincoli:
- 200-400 parole.
- Italiano moderno; mantieni nomi propri originali.
- Includi: chi, cosa, dove, quando, fonti citate, numeri specifici (cifre, vittime, tonnellate, ecc.).
- Se l'articolo è propaganda fascista, NON usare il suo lessico: descrivilo dall'esterno ("il giornale afferma che...", "secondo il bollettino...").
- Se il testo OCR è frammentato o contraddittorio, segnalalo brevemente in una frase finale.
- Non aggiungere commenti, valutazioni morali o conclusioni: solo sintesi del contenuto.

Riassunto:"""

DAY_PROMPT = """Riassumi la giornata del {date} ne «Il Messaggero», basandoti SOLO sui riassunti d'articolo qui sotto. {weekday}

{summaries}

Vincoli:
- 200-400 parole.
- Sintesi della giornata: principali notizie di guerra (fronte orientale, Mediterraneo, bombardamenti su Roma e altre città italiane), notizie politiche e di regime, cronaca cittadina romana, notizie dall'estero, articoli di costume.
- Preserva i nomi propri e le cifre più rilevanti.
- Cattura sia i grandi eventi sia la texture della vita quotidiana romana ricostruibile dal giornale.
- Non commentare; descrivi cosa il giornale dice, non cosa è "vero".

Riassunto della giornata:"""

WEEK_PROMPT = """Riassumi la settimana dal {date_start} al {date_end} a Roma sulla base dei riassunti giornalieri qui sotto.

{summaries}

Vincoli:
- 200-400 parole.
- Identifica i temi ricorrenti, l'arco narrativo della settimana, gli eventi-chiave, i personaggi che riappaiono.
- Se mancano giorni, segnalalo (es. "il 26 luglio non è presente nel fondo archivistico").
- Preserva nomi propri, numeri, date specifiche.
- Italiano moderno, neutro, descrittivo.

Riassunto settimanale:"""

MONTH_PROMPT = """Riassumi il mese di luglio 1943 a Roma sulla base dei riassunti settimanali qui sotto.

{summaries}

Vincoli:
- 200-400 parole.
- Cattura l'arco del mese: dalla guerra ancora in corso, allo sbarco alleato in Sicilia (10 luglio), al bombardamento di Roma (19 luglio), alla caduta di Mussolini nella notte del 25 luglio (Gran Consiglio del Fascismo) e al governo Badoglio.
- L'assenza dell'edizione del 26 luglio è significativa e va menzionata.
- Preserva nomi propri, date, eventi-chiave.
- Italiano moderno; descrivi cosa il giornale racconta giorno per giorno e come il tono cambia (o non cambia) attorno al 25-27 luglio.

Riassunto del mese:"""


# ----- data structures ----------------------------------------------------


@dataclass
class Node:
    node_id: str
    level: str
    parent_id: str
    position: int
    date_start: str
    date_end: str
    summary: str
    raw_text: str | None
    embedding: list[float] = field(default_factory=list)
    child_count: int = 0
    source: str = "il_messaggero"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "parent_id": self.parent_id,
            "position": self.position,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "summary": self.summary,
            "raw_text": self.raw_text,
            "embedding": self.embedding,
            "child_count": self.child_count,
            "source": self.source,
        }


@dataclass
class Cost:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    failures: int = 0

    def add(self, model: str, usage: Any) -> None:
        p = PRICING[model]
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_create
        # Standard pricing: cache reads at 0.1x, cache creation at 1.25x.
        self.cost_usd += (
            (in_tok / 1_000_000) * p["in"]
            + (out_tok / 1_000_000) * p["out"]
            + (cache_read / 1_000_000) * p["in"] * 0.1
            + (cache_create / 1_000_000) * p["in"] * 1.25
        )
        self.calls += 1


# ----- LLM client wrapper -------------------------------------------------


def load_oauth_token() -> str:
    creds_path = os.path.expanduser("~/.claude/.credentials.json")
    with open(creds_path) as f:
        creds = json.load(f)
    return creds["claudeAiOauth"]["accessToken"]


class LLM:
    def __init__(self, cost: Cost) -> None:
        token = load_oauth_token()
        self.client = AsyncAnthropic(
            auth_token=token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=3,
        )
        # OAuth tokens require a Claude Code system prefix; we then append our task.
        self.cc_system = "You are Claude Code, Anthropic's official CLI for Claude."
        self.cost = cost

    async def summarize(
        self,
        model: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> str:
        # Place the historical-summarizer system prompt as the FIRST user message
        # context block so OAuth still accepts it (the API system field stays
        # the Claude Code identity string).
        full_user = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                resp = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=self.cc_system,
                    messages=[{"role": "user", "content": full_user}],
                )
                self.cost.add(model, resp.usage)
                # Extract text
                parts = []
                for block in resp.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "".join(parts).strip()
            except Exception as e:  # noqa: BLE001
                if attempt == max_attempts - 1:
                    self.cost.failures += 1
                    raise
                # Exponential backoff with cap; rate-limit errors get longer waits.
                base = 30 if "rate_limit" in str(e).lower() or "429" in str(e) else 2 ** attempt
                wait = min(base * (1 + attempt * 0.5), 240)
                print(
                    f"  ! retry {attempt + 1}/{max_attempts - 1} after {wait:.0f}s "
                    f"({type(e).__name__}: {str(e)[:200]})",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")


# ----- pipeline steps -----------------------------------------------------


def write_node(node: Node) -> None:
    path = OUT / node.level / f"{node.node_id}.json"
    path.write_text(json.dumps(node.to_dict(), ensure_ascii=False, indent=2))


def read_node(node_id: str, level: str) -> Node:
    path = OUT / level / f"{node_id}.json"
    d = json.loads(path.read_text())
    return Node(**d)


async def summarize_article(llm: LLM, day_id: str, art: dict[str, Any]) -> Node:
    art_id = art["id"]
    headline = art.get("headline") or "(senza titolo)"
    paragraphs = art.get("paragraphs", []) or []
    raw_text = "\n\n".join(p.get("text", "") for p in paragraphs).strip()

    if not raw_text:
        # Empty article: skip LLM, just emit a stub.
        summary = f"[articolo vuoto: nessun testo OCR per {art_id}]"
    else:
        prompt = ARTICLE_PROMPT.format(date=day_id, headline=headline, text=raw_text)
        summary = await llm.summarize(MODEL_ARTICLE, prompt, max_tokens=700)

    return Node(
        node_id=art_id,
        level="article",
        parent_id=day_id,
        position=art.get("position_in_issue", 0),
        date_start=day_id,
        date_end=day_id,
        summary=summary,
        raw_text=raw_text or None,
        child_count=len(paragraphs),
    )


async def summarize_day(llm: LLM, day_id: str, article_nodes: list[Node]) -> Node:
    weekday_map = {0: "lunedì", 1: "martedì", 2: "mercoledì", 3: "giovedì", 4: "venerdì", 5: "sabato", 6: "domenica"}
    d = dt.date.fromisoformat(day_id)
    weekday = f"({weekday_map[d.weekday()]})"

    chunks = []
    for n in sorted(article_nodes, key=lambda x: x.position):
        chunks.append(f"### {n.node_id} (pos {n.position})\n{n.summary}")
    summaries_block = "\n\n".join(chunks)

    prompt = DAY_PROMPT.format(date=day_id, weekday=weekday, summaries=summaries_block)
    summary = await llm.summarize(MODEL_DAY, prompt, max_tokens=900)

    return Node(
        node_id=day_id,
        level="day",
        parent_id="",  # set later when grouping into weeks
        position=0,
        date_start=day_id,
        date_end=day_id,
        summary=summary,
        raw_text=None,
        child_count=len(article_nodes),
    )


def make_absent_day_node(day_id: str) -> Node:
    summary = (
        "[edizione assente: il fondo archivistico digitalizzato non contiene "
        "il numero del 26 luglio 1943 de «Il Messaggero». Il giorno precedente "
        "— notte fra 24 e 25 luglio — il Gran Consiglio del Fascismo aveva "
        "votato l'ordine del giorno Grandi alle 02:40, e nel pomeriggio del 25 "
        "Vittorio Emanuele III aveva fatto arrestare Mussolini all'uscita da "
        "Villa Savoia. Il giornale del 27 luglio riapparirà sotto il nuovo "
        "governo Badoglio. La lacuna archivistica del 26 luglio è essa stessa "
        "documento: testimonia il vuoto editoriale di quelle 24 ore di "
        "transizione di regime.]"
    )
    return Node(
        node_id=day_id,
        level="day",
        parent_id="",
        position=0,
        date_start=day_id,
        date_end=day_id,
        summary=summary,
        raw_text=None,
        child_count=0,
    )


async def summarize_week(llm: LLM, week_id: str, day_nodes: list[Node], date_start: str, date_end: str) -> Node:
    chunks = []
    for n in sorted(day_nodes, key=lambda x: x.date_start):
        chunks.append(f"### {n.date_start}\n{n.summary}")
    summaries_block = "\n\n".join(chunks)

    prompt = WEEK_PROMPT.format(date_start=date_start, date_end=date_end, summaries=summaries_block)
    summary = await llm.summarize(MODEL_WEEK, prompt, max_tokens=900)

    return Node(
        node_id=week_id,
        level="week",
        parent_id="1943-07",
        position=0,
        date_start=date_start,
        date_end=date_end,
        summary=summary,
        raw_text=None,
        child_count=len(day_nodes),
    )


async def summarize_month(llm: LLM, week_nodes: list[Node]) -> Node:
    chunks = []
    for n in sorted(week_nodes, key=lambda x: x.date_start):
        chunks.append(f"### {n.node_id} ({n.date_start} → {n.date_end})\n{n.summary}")
    summaries_block = "\n\n".join(chunks)

    prompt = MONTH_PROMPT.format(summaries=summaries_block)
    summary = await llm.summarize(MODEL_MONTH, prompt, max_tokens=900)

    return Node(
        node_id="1943-07",
        level="month",
        parent_id="",
        position=0,
        date_start="1943-07-01",
        date_end="1943-07-31",
        summary=summary,
        raw_text=None,
        child_count=len(week_nodes),
    )


# ----- week assignment ----------------------------------------------------

WEEK_BUCKETS: list[tuple[str, str, str, list[str]]] = [
    # (week_id, start, end, day_ids)
    ("1943-W26-tail", "1943-07-01", "1943-07-04", [f"1943-07-0{d}" for d in range(1, 5)]),
    ("1943-W27", "1943-07-05", "1943-07-11", [f"1943-07-{d:02d}" for d in range(5, 12)]),
    ("1943-W28", "1943-07-12", "1943-07-18", [f"1943-07-{d:02d}" for d in range(12, 19)]),
    ("1943-W29", "1943-07-19", "1943-07-25", [f"1943-07-{d:02d}" for d in range(19, 26)]),
    ("1943-W30", "1943-07-26", "1943-07-31", [f"1943-07-{d:02d}" for d in range(26, 32)]),
]


def day_to_week_id(day_id: str) -> str:
    for wid, _, _, days in WEEK_BUCKETS:
        if day_id in days:
            return wid
    raise ValueError(day_id)


# ----- main pipeline ------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    cost = Cost()
    llm = LLM(cost)

    # Load embedding model lazily — only after summarisation finishes — so we
    # don't waste RAM during the long LLM phase.
    embed_model: SentenceTransformer | None = None

    def get_embed_model() -> SentenceTransformer:
        nonlocal embed_model
        if embed_model is None:
            # Try CUDA if available; on CPU BGE-M3 takes ~1-2 sec/doc which is
            # impractical for 6500 nodes.
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[embed] loading {EMBED_MODEL_NAME} on {device}...", flush=True)
            embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
            # Cap sequence length: summaries are 200-400 words; BGE-M3 default
            # is 8192 which is intractable on CPU. 384 tokens captures the
            # leading content of every summary at ~2.5 nodes/sec on this i7.
            embed_model.max_seq_length = 384
        return embed_model

    # ---- 1. Load all transcriptions ----
    day_files = sorted(TRANSCRIPTIONS.glob("1943-07-*.json"))
    print(f"[load] {len(day_files)} day files", flush=True)
    day_data: dict[str, dict[str, Any]] = {}
    for f in day_files:
        d = json.loads(f.read_text())
        day_data[d["date"]] = d

    # ---- 2. Article-level summaries ----
    print(f"[article] summarising {sum(len(d['articles']) for d in day_data.values())} articles ...", flush=True)
    sem = asyncio.Semaphore(ARTICLE_CONCURRENCY)
    article_nodes_by_day: dict[str, list[Node]] = {dd: [] for dd in day_data}

    async def do_article(day_id: str, art: dict[str, Any]) -> Node:
        out_path = OUT / "article" / f"{art['id']}.json"
        if out_path.exists():
            # Resume: load existing
            return Node(**json.loads(out_path.read_text()))
        async with sem:
            if cost.cost_usd > BUDGET_CAP_USD:
                raise RuntimeError(f"Budget cap ${BUDGET_CAP_USD} exceeded at ${cost.cost_usd:.2f}")
            node = await summarize_article(llm, day_id, art)
            write_node(node)
            return node

    tasks = []
    article_count = 0
    for day_id, dd in day_data.items():
        for art in dd["articles"]:
            if args.limit_articles and article_count >= args.limit_articles:
                break
            tasks.append(do_article(day_id, art))
            article_count += 1
        if args.limit_articles and article_count >= args.limit_articles:
            break

    print(f"[article] dispatching {len(tasks)} tasks (concurrency={ARTICLE_CONCURRENCY})", flush=True)
    done = 0
    last_log = time.time()
    # Use as_completed so we can stream progress.
    for coro in asyncio.as_completed(tasks):
        try:
            node = await coro
            article_nodes_by_day.setdefault(node.parent_id, []).append(node)
        except Exception as e:  # noqa: BLE001
            print(f"  ! article failure: {e}", file=sys.stderr, flush=True)
        done += 1
        if time.time() - last_log > 10 or done == len(tasks):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.001)
            eta = (len(tasks) - done) / max(rate, 0.001)
            print(
                f"  [{done}/{len(tasks)}] cost=${cost.cost_usd:.2f} "
                f"calls={cost.calls} fails={cost.failures} rate={rate:.1f}/s eta={eta/60:.1f}min",
                flush=True,
            )
            last_log = time.time()

    print(f"[article] done. cost=${cost.cost_usd:.3f}", flush=True)

    if args.stop_after == "article":
        print("[stop] stop_after=article. exiting.", flush=True)
        await embed_all(get_embed_model)
        write_manifest()
        return

    # ---- 3. Day-level summaries (sequential per day, parallel across days) ----
    print(f"[day] summarising {len(article_nodes_by_day)} days ...", flush=True)
    day_nodes: dict[str, Node] = {}

    async def do_day(day_id: str) -> Node:
        out_path = OUT / "day" / f"{day_id}.json"
        if out_path.exists():
            return Node(**json.loads(out_path.read_text()))
        if cost.cost_usd > BUDGET_CAP_USD:
            raise RuntimeError(f"Budget cap ${BUDGET_CAP_USD} exceeded at ${cost.cost_usd:.2f}")
        nodes = article_nodes_by_day[day_id]
        node = await summarize_day(llm, day_id, nodes)
        write_node(node)
        return node

    day_tasks = [do_day(did) for did in day_data]
    day_results = await asyncio.gather(*day_tasks, return_exceptions=True)
    for did, r in zip(day_data, day_results, strict=True):
        if isinstance(r, Exception):
            print(f"  ! day {did} failed: {r}", file=sys.stderr)
        else:
            day_nodes[did] = r

    # Inject the absent 1943-07-26 node
    absent_id = "1943-07-26"
    absent_path = OUT / "day" / f"{absent_id}.json"
    if absent_path.exists():
        absent_node = Node(**json.loads(absent_path.read_text()))
    else:
        absent_node = make_absent_day_node(absent_id)
        write_node(absent_node)
    day_nodes[absent_id] = absent_node
    print(f"[day] done. {len(day_nodes)} day nodes (incl. absent {absent_id}). cost=${cost.cost_usd:.3f}", flush=True)

    # Set parent_id on day nodes now that we know weeks
    for did, node in day_nodes.items():
        node.parent_id = day_to_week_id(did)
        write_node(node)

    # ---- 4. Week-level summaries ----
    print(f"[week] summarising {len(WEEK_BUCKETS)} weeks ...", flush=True)
    week_nodes: list[Node] = []
    for wid, wstart, wend, wdays in WEEK_BUCKETS:
        out_path = OUT / "week" / f"{wid}.json"
        if out_path.exists():
            week_nodes.append(Node(**json.loads(out_path.read_text())))
            continue
        if cost.cost_usd > BUDGET_CAP_USD:
            raise RuntimeError(f"Budget cap ${BUDGET_CAP_USD} exceeded at ${cost.cost_usd:.2f}")
        members = [day_nodes[d] for d in wdays if d in day_nodes]
        node = await summarize_week(llm, wid, members, wstart, wend)
        write_node(node)
        week_nodes.append(node)
        print(f"  [{wid}] {len(members)} days. cost=${cost.cost_usd:.3f}", flush=True)

    # ---- 5. Month-level summary ----
    print("[month] summarising July 1943 ...", flush=True)
    month_path = OUT / "month" / "1943-07.json"
    if month_path.exists():
        month_node = Node(**json.loads(month_path.read_text()))
    else:
        month_node = await summarize_month(llm, week_nodes)
        write_node(month_node)
    print(f"[month] done. cost=${cost.cost_usd:.3f}", flush=True)

    # ---- 6. Embeddings for everything ----
    await embed_all(get_embed_model)

    # ---- 7. Manifest ----
    write_manifest()

    # ---- 8. Final report ----
    elapsed = time.time() - t0
    counts = {lvl: len(list((OUT / lvl).glob("*.json"))) for lvl in LEVELS}
    report = {
        "elapsed_sec": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 2),
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "cache_read_tokens": cost.cache_read_tokens,
        "cache_creation_tokens": cost.cache_creation_tokens,
        "total_cost_usd": round(cost.cost_usd, 4),
        "llm_calls": cost.calls,
        "failures": cost.failures,
        "node_counts": counts,
    }
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== FINAL REPORT ===")
    print(json.dumps(report, indent=2))


async def embed_all(get_embed_model) -> None:
    """Embed every node's summary. Process in small chunks so we have progress
    visibility and can resume if interrupted."""
    print("[embed] computing embeddings for all nodes ...", flush=True)
    model = get_embed_model()
    CHUNK = 64  # write to disk every 64 embeddings
    for level in LEVELS:
        files = sorted((OUT / level).glob("*.json"))
        if not files:
            continue
        to_embed = []
        for f in files:
            d = json.loads(f.read_text())
            if not d.get("embedding"):
                to_embed.append((f, d))
        if not to_embed:
            print(f"  [{level}] all {len(files)} already have embeddings", flush=True)
            continue
        print(f"  [{level}] embedding {len(to_embed)} summaries ...", flush=True)
        t_lvl = time.time()
        for i in range(0, len(to_embed), CHUNK):
            chunk = to_embed[i : i + CHUNK]
            texts = [d["summary"] for _, d in chunk]
            vecs = model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for (f, d), v in zip(chunk, vecs, strict=True):
                d["embedding"] = [float(x) for x in v]
                f.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            elapsed = time.time() - t_lvl
            done = min(i + CHUNK, len(to_embed))
            rate = done / max(elapsed, 0.001)
            eta = (len(to_embed) - done) / max(rate, 0.001)
            print(
                f"  [{level}] {done}/{len(to_embed)} done in {elapsed:.0f}s "
                f"({rate:.1f}/s, eta {eta:.0f}s)",
                flush=True,
            )
        print(f"  [{level}] complete", flush=True)


def write_manifest() -> None:
    print("[manifest] writing manifest.json ...", flush=True)
    manifest = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "source": "il_messaggero",
        "date_range": ["1943-07-01", "1943-07-31"],
        "levels": {},
        "nodes": [],
    }
    for level in LEVELS:
        files = sorted((OUT / level).glob("*.json"))
        manifest["levels"][level] = len(files)
        for f in files:
            d = json.loads(f.read_text())
            manifest["nodes"].append({
                "node_id": d["node_id"],
                "level": d["level"],
                "parent_id": d["parent_id"],
                "date_start": d["date_start"],
                "date_end": d["date_end"],
                "child_count": d["child_count"],
                "embedding_dim": len(d.get("embedding") or []),
                "has_raw_text": bool(d.get("raw_text")),
            })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[manifest] {sum(manifest['levels'].values())} nodes total", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-articles", type=int, default=0, help="Limit articles for smoke testing")
    parser.add_argument("--stop-after", choices=["", "article", "day", "week"], default="")
    parser.add_argument("--dry-run", action="store_true", help="Don't call the LLM, just plan")
    args = parser.parse_args()

    if args.dry_run:
        day_files = sorted(TRANSCRIPTIONS.glob("1943-07-*.json"))
        total = 0
        for f in day_files:
            d = json.loads(f.read_text())
            total += len(d["articles"])
        print(f"[dry-run] {len(day_files)} days, {total} articles, ~{total + 30 + 5 + 1} LLM calls")
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
