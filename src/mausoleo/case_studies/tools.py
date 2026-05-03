"""Tool implementations for the researcher agent.

Two toolsets are exposed:

- ``MAUSOLEO_TOOLS``: hierarchical-index access (root, node, children, parent,
  text, stats, search semantic, search text, search hybrid). Backed directly
  by ClickHouse via ``clickhouse_connect`` so the agent does not need the
  HTTP layer up.
- ``BASELINE_TOOLS``: only ``baseline_search`` (BM25-style FTS over the
  ``documents`` table) and ``read_article``. The baseline gets no access
  to the ``nodes`` hierarchy by construction.

Each tool returns a JSON-serialisable dict. Long string fields are truncated
to keep the agent's context manageable; the agent can drill into ``text``
explicitly when it needs the full primary source.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import typing as tp

import clickhouse_connect

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_CLIENT: tp.Any = None


def get_client() -> tp.Any:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = clickhouse_connect.get_client(
            host="127.0.0.1", port=8123, database="default"
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim(s: str | None, n: int = 600) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _serialize_node(row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    out = {
        "node_id": row.get("node_id"),
        "level": row.get("level"),
        "parent_id": row.get("parent_id"),
        "date_start": _date(row.get("date_start")),
        "date_end": _date(row.get("date_end")),
        "summary": row.get("summary") or "",
        "child_count": int(row.get("child_count", 0) or 0),
    }
    return out


def _date(v: tp.Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return str(v)


# ---------------------------------------------------------------------------
# Mausoleo tools (nodes table)
# ---------------------------------------------------------------------------

def mausoleo_root() -> dict[str, tp.Any]:
    cli = get_client()
    rows = cli.query(
        "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count "
        "FROM nodes WHERE node_id = 'archive' LIMIT 1"
    ).named_results()
    rows = list(rows)
    if not rows:
        rows = list(
            cli.query(
                "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count "
                "FROM nodes ORDER BY level DESC, date_start ASC LIMIT 1"
            ).named_results()
        )
    if not rows:
        return {"error": "empty index"}
    return _serialize_node(rows[0])


def mausoleo_node(node_id: str) -> dict[str, tp.Any]:
    cli = get_client()
    rows = list(
        cli.query(
            "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count "
            "FROM nodes WHERE node_id = {nid:String} LIMIT 1",
            parameters={"nid": node_id},
        ).named_results()
    )
    if not rows:
        return {"error": f"node {node_id} not found"}
    return _serialize_node(rows[0])


def mausoleo_children(node_id: str, limit: int = 60) -> dict[str, tp.Any]:
    cli = get_client()
    rows = list(
        cli.query(
            "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count "
            "FROM nodes WHERE parent_id = {nid:String} "
            "ORDER BY position ASC, date_start ASC LIMIT {lim:UInt32}",
            parameters={"nid": node_id, "lim": int(limit)},
        ).named_results()
    )
    return {
        "parent_id": node_id,
        "count": len(rows),
        "children": [_serialize_node(r) for r in rows],
    }


def mausoleo_parent(node_id: str) -> dict[str, tp.Any]:
    cli = get_client()
    rows = list(
        cli.query(
            "SELECT parent_id FROM nodes WHERE node_id = {nid:String} LIMIT 1",
            parameters={"nid": node_id},
        ).named_results()
    )
    if not rows:
        return {"error": f"node {node_id} not found"}
    pid = rows[0]["parent_id"]
    if not pid:
        return {"node_id": node_id, "parent": None}
    return mausoleo_node(pid)


def mausoleo_text(node_id: str) -> dict[str, tp.Any]:
    cli = get_client()
    head = list(
        cli.query(
            "SELECT level, raw_text FROM nodes WHERE node_id = {nid:String} LIMIT 1",
            parameters={"nid": node_id},
        ).named_results()
    )
    if not head:
        return {"error": f"node {node_id} not found"}
    if head[0]["level"] == "paragraph" and head[0].get("raw_text"):
        return {"node_id": node_id, "level": "paragraph", "text": head[0]["raw_text"]}
    # Walk to descendant paragraphs.
    frontier = [node_id]
    pieces: list[tuple[dt.date, int, str]] = []
    visited: set[str] = set()
    while frontier:
        rows = list(
            cli.query(
                "SELECT node_id, level, position, date_start, raw_text FROM nodes "
                "WHERE parent_id IN ({ids:Array(String)})",
                parameters={"ids": frontier},
            ).named_results()
        )
        next_f: list[str] = []
        for r in rows:
            if r["node_id"] in visited:
                continue
            visited.add(r["node_id"])
            if r["level"] == "paragraph" and r.get("raw_text"):
                pieces.append((r["date_start"], int(r["position"]), r["raw_text"]))
            else:
                next_f.append(r["node_id"])
        frontier = next_f
    pieces.sort()
    text = "\n\n".join(p[2] for p in pieces)
    return {
        "node_id": node_id,
        "level": head[0]["level"],
        "paragraph_count": len(pieces),
        "text": text,
    }


def mausoleo_stats() -> dict[str, tp.Any]:
    cli = get_client()
    by_level = list(
        cli.query(
            "SELECT level, count() AS n, min(date_start) AS d_min, max(date_end) AS d_max "
            "FROM nodes GROUP BY level ORDER BY level"
        ).named_results()
    )
    return {
        "by_level": [
            {
                "level": r["level"],
                "count": int(r["n"]),
                "date_min": _date(r["d_min"]),
                "date_max": _date(r["d_max"]),
            }
            for r in by_level
        ],
    }


def _text_search_summary(
    query: str,
    level: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 15,
) -> dict[str, tp.Any]:
    cli = get_client()
    where: list[str] = ["positionCaseInsensitive(summary, {q:String}) > 0"]
    params: dict[str, tp.Any] = {"q": query, "lim": int(limit)}
    if level:
        where.append("level = {lvl:String}")
        params["lvl"] = level
    if date_from:
        where.append("date_end >= {df:Date32}")
        params["df"] = dt.date.fromisoformat(date_from)
    if date_to:
        where.append("date_start <= {dt:Date32}")
        params["dt"] = dt.date.fromisoformat(date_to)
    sql = (
        "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count "
        "FROM nodes WHERE " + " AND ".join(where) +
        " ORDER BY level ASC, date_start ASC, position ASC LIMIT {lim:UInt32}"
    )
    rows = list(cli.query(sql, parameters=params).named_results())
    return {
        "query": query,
        "count": len(rows),
        "results": [_serialize_node(r) for r in rows],
    }


def mausoleo_search_text(query: str, level: str | None = None,
                          date_from: str | None = None, date_to: str | None = None,
                          limit: int = 15) -> dict[str, tp.Any]:
    return _text_search_summary(query, level=level, date_from=date_from,
                                date_to=date_to, limit=limit)


# Semantic + hybrid require an embedding model. We use the same
# paraphrase-multilingual-MiniLM-L12-v2 fallback that built the index. To
# avoid loading 500 MB of model weights into the agent harness, semantic and
# hybrid are exposed as L2-distance brute-force queries against pre-stored
# embeddings only when an embedding model is loaded; otherwise they fall
# back to the text-search path. This is documented as a methodology note in
# RUNLOG: under the budget cap we did not load BGE-M3 / MiniLM during the
# trials, so semantic and hybrid resolve to text search at run time. The
# agent is told this in its system prompt so it does not assume otherwise.

_EMBED_MODEL: tp.Any = None


def _load_embedder() -> tp.Any | None:
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        _EMBED_MODEL.max_seq_length = 384
        return _EMBED_MODEL
    except Exception:
        return None


def mausoleo_search_semantic(query: str, level: str | None = None,
                              date_from: str | None = None, date_to: str | None = None,
                              limit: int = 15) -> dict[str, tp.Any]:
    cli = get_client()
    embedder = _load_embedder()
    if embedder is None:
        # Fallback to text-search; documented in RUNLOG.
        return mausoleo_search_text(query, level=level, date_from=date_from,
                                    date_to=date_to, limit=limit)
    vec = embedder.encode(query, normalize_embeddings=True).tolist()
    where: list[str] = ["1=1"]
    params: dict[str, tp.Any] = {"q": vec, "lim": int(limit)}
    if level:
        where.append("level = {lvl:String}")
        params["lvl"] = level
    if date_from:
        where.append("date_end >= {df:Date32}")
        params["df"] = dt.date.fromisoformat(date_from)
    if date_to:
        where.append("date_start <= {dt:Date32}")
        params["dt"] = dt.date.fromisoformat(date_to)
    sql = (
        "SELECT node_id, level, parent_id, date_start, date_end, summary, child_count, "
        "L2Distance(embedding, {q:Array(Float32)}) AS distance "
        "FROM nodes WHERE " + " AND ".join(where) +
        " ORDER BY distance ASC LIMIT {lim:UInt32}"
    )
    rows = list(cli.query(sql, parameters=params).named_results())
    return {
        "query": query,
        "count": len(rows),
        "results": [_serialize_node(r) for r in rows],
    }


def mausoleo_search_hybrid(query: str, level: str | None = None,
                            date_from: str | None = None, date_to: str | None = None,
                            limit: int = 15) -> dict[str, tp.Any]:
    """RRF over semantic + text. Falls back to text-only if no embedder."""
    sem = mausoleo_search_semantic(query, level=level, date_from=date_from,
                                    date_to=date_to, limit=limit * 4)
    txt = mausoleo_search_text(query, level=level, date_from=date_from,
                                date_to=date_to, limit=limit * 4)
    score: dict[str, float] = {}
    rec: dict[str, dict[str, tp.Any]] = {}
    for r in sem.get("results", []):
        nid = r["node_id"]
        score[nid] = score.get(nid, 0.0) + 1.0 / (60 + len(score) + 1)
        rec[nid] = r
    for r in txt.get("results", []):
        nid = r["node_id"]
        score[nid] = score.get(nid, 0.0) + 1.0 / (60 + len(score) + 1)
        rec.setdefault(nid, r)
    out = sorted(rec.values(), key=lambda r: -score[r["node_id"]])[:limit]
    return {"query": query, "count": len(out), "results": out}


# ---------------------------------------------------------------------------
# Baseline tools (documents table; flat BM25-style FTS)
# ---------------------------------------------------------------------------

# Italian + a few English / WW2-relevant stopwords for the BM25 scorer.
_STOP = set(
    """
a ad agli ai al alla alle allo ai anche ancora avere avete aveva avevano avevamo
avete avranno avrei avresti avremmo avremo c che chi ci cui da dai dal dalla
dalle dallo degli dei del della delle dello di e ed essi essere fra ha hanno
ho i il in io l la le lo loro ma mi mio mia ne nei nel nella nelle nello noi
non nostre nostri nostro o o per perchè perche piu più poi però se senza si sia
siamo siate siete sono su sua sue suo suoi sul sulla sulle sullo te ti tra tu
tue tuo tuoi tutti tutto un una uno unico unico ve voi vostre vostri vostro y
to of in for and the a an or with by from as is are be been being was were
""".split()
)

_TOK_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def _tokenise(s: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(s) if len(t) > 1 and t.lower() not in _STOP]


def _bm25_search_python(
    query: str, date_from: str | None, date_to: str | None, limit: int
) -> dict[str, tp.Any]:
    """In-Python BM25 over the documents table.

    Loaded once (memoised) — the corpus is ~1M tokens, fits comfortably in RAM.
    Stronger than ClickHouse's positionCaseInsensitive baseline because it
    actually scores by IDF-weighted overlap.
    """
    cli = get_client()
    docs = _bm25_corpus()
    qterms = _tokenise(query)
    if not qterms:
        return {"query": query, "count": 0, "results": []}
    df_from = dt.date.fromisoformat(date_from) if date_from else None
    df_to = dt.date.fromisoformat(date_to) if date_to else None
    k1 = 1.5
    b = 0.75
    avgdl = docs["avgdl"]
    N = docs["N"]
    df = docs["df"]
    idf = {t: math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in set(qterms)}
    scored: list[tuple[float, int]] = []
    for i, doc in enumerate(docs["rows"]):
        if df_from and doc["date"] < df_from:
            continue
        if df_to and doc["date"] > df_to:
            continue
        s = 0.0
        dl = doc["dl"]
        tf = doc["tf"]
        for t in qterms:
            if t not in tf:
                continue
            f = tf[t]
            denom = f + k1 * (1 - b + b * dl / avgdl)
            s += idf[t] * (f * (k1 + 1)) / denom
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:limit]:
        d = docs["rows"][i]
        out.append({
            "article_id": d["article_id"],
            "date": d["date"].isoformat(),
            "headline": d["headline"],
            "snippet": _trim(d["text"], 220),
            "score": round(score, 4),
        })
    return {"query": query, "count": len(out), "results": out}


_DOCS_CACHE: dict[str, tp.Any] | None = None


def _bm25_corpus() -> dict[str, tp.Any]:
    global _DOCS_CACHE
    if _DOCS_CACHE is not None:
        return _DOCS_CACHE
    cli = get_client()
    rows = list(
        cli.query(
            "SELECT article_id, date, headline, text FROM documents "
            "WHERE date >= '1943-07-01' AND date <= '1943-07-31' "
            "ORDER BY date ASC, article_id ASC"
        ).named_results()
    )
    docs: list[dict[str, tp.Any]] = []
    df: dict[str, int] = {}
    total_dl = 0
    for r in rows:
        toks = _tokenise(r["text"] + " " + (r["headline"] or ""))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df[t] = df.get(t, 0) + 1
        total_dl += len(toks)
        docs.append({
            "article_id": r["article_id"],
            "date": r["date"] if isinstance(r["date"], dt.date) else dt.date.fromisoformat(str(r["date"])),
            "headline": r.get("headline", ""),
            "text": r["text"],
            "tf": tf,
            "dl": len(toks),
        })
    _DOCS_CACHE = {
        "rows": docs,
        "df": df,
        "N": len(docs),
        "avgdl": (total_dl / max(len(docs), 1)),
    }
    return _DOCS_CACHE


def baseline_search(query: str, date_from: str | None = None,
                     date_to: str | None = None, limit: int = 15) -> dict[str, tp.Any]:
    """BM25 over the flat documents table. Returns ranked article snippets."""
    return _bm25_search_python(query, date_from, date_to, limit)


def baseline_read_article(article_id: str) -> dict[str, tp.Any]:
    cli = get_client()
    rows = list(
        cli.query(
            "SELECT article_id, date, headline, text, page_span FROM documents "
            "WHERE article_id = {aid:String} LIMIT 1",
            parameters={"aid": article_id},
        ).named_results()
    )
    if not rows:
        return {"error": f"article {article_id} not found"}
    r = rows[0]
    return {
        "article_id": r["article_id"],
        "date": r["date"].isoformat() if isinstance(r["date"], dt.date) else str(r["date"]),
        "headline": r.get("headline", ""),
        "page_span": r.get("page_span", ""),
        "text": r["text"],
    }


# ---------------------------------------------------------------------------
# Tool dispatch tables (Anthropic tool-use schema)
# ---------------------------------------------------------------------------

MAUSOLEO_TOOL_SCHEMAS: list[dict[str, tp.Any]] = [
    {
        "name": "root",
        "description": "Get the archive root node — the highest-level summary entry point.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "node",
        "description": "Get a specific node's metadata + summary by id (e.g. '1943-07', '1943-07-25', '1943-07-25_a012').",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "children",
        "description": "List the direct children of a node (e.g. children of a month = days; children of a day = articles).",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "limit": {"type": "integer", "default": 60},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "parent",
        "description": "Walk up the tree by one level.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "text",
        "description": "Get raw text. For paragraph nodes, returns the paragraph; for higher levels, reconstructs full text from descendants.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "stats",
        "description": "Index statistics: number of nodes per level, date range covered.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_semantic",
        "description": (
            "Semantic vector search over node summaries. (Note: in this evaluation "
            "configuration the embedding model was not loaded due to budget; this "
            "tool falls back to text search at runtime.)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "level": {"type": "string", "description": "filter by level: paragraph|article|day|week|month"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "default": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_text",
        "description": "Substring-match search over node summaries (case-insensitive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "level": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "limit": {"type": "integer", "default": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_hybrid",
        "description": "RRF combination of semantic + text rankings over node summaries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "level": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "limit": {"type": "integer", "default": 15},
            },
            "required": ["query"],
        },
    },
]

BASELINE_TOOL_SCHEMAS: list[dict[str, tp.Any]] = [
    {
        "name": "baseline_search",
        "description": "BM25 ranked search over the flat article corpus (Il Messaggero July 1943, ~6480 articles). Returns headline + 220-char snippet + score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Italian or English keywords."},
                "date_from": {"type": "string", "description": "YYYY-MM-DD inclusive."},
                "date_to": {"type": "string", "description": "YYYY-MM-DD inclusive."},
                "limit": {"type": "integer", "default": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_article",
        "description": "Fetch the full text of a single article by id (e.g. '1943-07-25_a012').",
        "input_schema": {
            "type": "object",
            "properties": {"article_id": {"type": "string"}},
            "required": ["article_id"],
        },
    },
]


def dispatch_mausoleo(name: str, kwargs: dict[str, tp.Any]) -> dict[str, tp.Any]:
    if name == "root":
        return mausoleo_root()
    if name == "node":
        return mausoleo_node(kwargs["node_id"])
    if name == "children":
        return mausoleo_children(kwargs["node_id"], int(kwargs.get("limit", 60)))
    if name == "parent":
        return mausoleo_parent(kwargs["node_id"])
    if name == "text":
        out = mausoleo_text(kwargs["node_id"])
        # Cap text returned to keep ctx manageable.
        if "text" in out and isinstance(out["text"], str) and len(out["text"]) > 12000:
            out["text"] = out["text"][:12000] + "\n\n[... truncated]"
            out["truncated"] = True
        return out
    if name == "stats":
        return mausoleo_stats()
    if name == "search_semantic":
        return mausoleo_search_semantic(
            kwargs["query"], level=kwargs.get("level"),
            date_from=kwargs.get("date_from"), date_to=kwargs.get("date_to"),
            limit=int(kwargs.get("limit", 15)))
    if name == "search_text":
        return mausoleo_search_text(
            kwargs["query"], level=kwargs.get("level"),
            date_from=kwargs.get("date_from"), date_to=kwargs.get("date_to"),
            limit=int(kwargs.get("limit", 15)))
    if name == "search_hybrid":
        return mausoleo_search_hybrid(
            kwargs["query"], level=kwargs.get("level"),
            date_from=kwargs.get("date_from"), date_to=kwargs.get("date_to"),
            limit=int(kwargs.get("limit", 15)))
    return {"error": f"unknown tool: {name}"}


def dispatch_baseline(name: str, kwargs: dict[str, tp.Any]) -> dict[str, tp.Any]:
    if name == "baseline_search":
        return baseline_search(
            kwargs["query"],
            date_from=kwargs.get("date_from"),
            date_to=kwargs.get("date_to"),
            limit=int(kwargs.get("limit", 15)),
        )
    if name == "read_article":
        out = baseline_read_article(kwargs["article_id"])
        if "text" in out and isinstance(out["text"], str) and len(out["text"]) > 12000:
            out["text"] = out["text"][:12000] + "\n\n[... truncated]"
            out["truncated"] = True
        return out
    return {"error": f"unknown tool: {name}"}
