"""HTTP endpoints for the Mausoleo search API.

All endpoints return plain ``dict`` payloads (FastAPI auto-serializes them).
The CLI consumes the JSON directly; we keep the schema simple and stable.
"""
from __future__ import annotations

import datetime as dt
import typing as tp

import fastapi as fa
import pydantic as pyd

from mausoleo.server import search as search_mod
from mausoleo.server.db import Db
from mausoleo.server.embed import default_embedder

router = fa.APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(pyd.BaseModel):
    query: str
    level: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    limit: int = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db(request: fa.Request) -> Db:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise fa.HTTPException(status_code=500, detail="db not initialised")
    assert isinstance(db, Db)
    return db


def _serialize_node(row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    out = dict(row)
    for k in ("date_start", "date_end"):
        v = out.get(k)
        if isinstance(v, (dt.date, dt.datetime)):
            out[k] = v.isoformat()
    # don't ship embeddings over HTTP
    out.pop("embedding", None)
    return out


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/root")
async def root_node(request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    # Prefer a real archive node; fall back to the highest-level node we have.
    row = db.query_one(
        "SELECT * FROM nodes WHERE node_id = 'archive' LIMIT 1"
    )
    if not row:
        row = db.query_one(
            "SELECT * FROM nodes ORDER BY level DESC, date_start ASC LIMIT 1"
        )
    if not row:
        raise fa.HTTPException(status_code=404, detail="empty index")
    return _serialize_node(row)


@router.get("/nodes/{node_id}")
async def get_node(node_id: str, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    row = db.query_one(
        "SELECT * FROM nodes WHERE node_id = {nid:String} LIMIT 1",
        {"nid": node_id},
    )
    if not row:
        raise fa.HTTPException(status_code=404, detail=f"node {node_id} not found")
    return _serialize_node(row)


@router.get("/nodes/{node_id}/children")
async def get_children(
    node_id: str,
    request: fa.Request,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, tp.Any]:
    db = _get_db(request)
    rows = db.query(
        "SELECT * FROM nodes WHERE parent_id = {nid:String} "
        "ORDER BY position ASC, date_start ASC "
        "LIMIT {lim:UInt32} OFFSET {off:UInt32}",
        {"nid": node_id, "lim": int(limit), "off": int(offset)},
    )
    total_row = db.query_one(
        "SELECT count() AS c FROM nodes WHERE parent_id = {nid:String}",
        {"nid": node_id},
    )
    total = int(total_row["c"]) if total_row else 0
    return {
        "parent_id": node_id,
        "offset": offset,
        "limit": limit,
        "total": total,
        "children": [_serialize_node(r) for r in rows],
    }


@router.get("/nodes/{node_id}/parent")
async def get_parent(node_id: str, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    me = db.query_one(
        "SELECT parent_id FROM nodes WHERE node_id = {nid:String} LIMIT 1",
        {"nid": node_id},
    )
    if not me:
        raise fa.HTTPException(status_code=404, detail=f"node {node_id} not found")
    parent_id = me["parent_id"]
    if not parent_id:
        return {"parent": None, "node_id": node_id}
    row = db.query_one(
        "SELECT * FROM nodes WHERE node_id = {nid:String} LIMIT 1",
        {"nid": parent_id},
    )
    if not row:
        raise fa.HTTPException(status_code=404, detail=f"parent {parent_id} not found")
    return _serialize_node(row)


@router.get("/nodes/{node_id}/text")
async def get_text(node_id: str, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    me = db.query_one(
        "SELECT level, raw_text FROM nodes WHERE node_id = {nid:String} LIMIT 1",
        {"nid": node_id},
    )
    if not me:
        raise fa.HTTPException(status_code=404, detail=f"node {node_id} not found")
    if me["level"] == "paragraph" and me.get("raw_text"):
        return {"node_id": node_id, "level": "paragraph", "text": me["raw_text"]}

    # Walk down to all descendant paragraphs and concatenate their raw_text.
    # We drill level-by-level; the tree is shallow (max 6 levels).
    frontier = [node_id]
    pieces: list[tuple[dt.date, int, str, str]] = []
    visited: set[str] = set()
    while frontier:
        rows = db.query(
            "SELECT node_id, level, parent_id, position, date_start, raw_text "
            "FROM nodes WHERE parent_id IN ({ids:Array(String)})",
            {"ids": frontier},
        )
        next_frontier: list[str] = []
        for r in rows:
            nid = r["node_id"]
            if nid in visited:
                continue
            visited.add(nid)
            if r["level"] == "paragraph" and r.get("raw_text"):
                pieces.append(
                    (
                        r["date_start"],
                        int(r["position"]),
                        nid,
                        r["raw_text"],
                    )
                )
            else:
                next_frontier.append(nid)
        frontier = next_frontier
    pieces.sort()
    text = "\n\n".join(p[3] for p in pieces)
    return {
        "node_id": node_id,
        "level": me["level"],
        "paragraph_count": len(pieces),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@router.post("/search/semantic")
async def search_semantic(req: SearchRequest, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    embedder = default_embedder()
    vec = embedder.encode(req.query)
    rows = search_mod.semantic_search(
        db,
        vec,
        level=req.level,
        date_start=req.date_start,
        date_end=req.date_end,
        limit=req.limit,
    )
    return {"query": req.query, "mode": "semantic", "results": [_serialize_node(r) for r in rows]}


@router.post("/search/text")
async def search_text(req: SearchRequest, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    rows = search_mod.text_search(
        db,
        req.query,
        level=req.level,
        date_start=req.date_start,
        date_end=req.date_end,
        limit=req.limit,
    )
    return {"query": req.query, "mode": "text", "results": [_serialize_node(r) for r in rows]}


@router.post("/search/hybrid")
async def search_hybrid(req: SearchRequest, request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    embedder = default_embedder()
    vec = embedder.encode(req.query)
    rows = search_mod.hybrid_search(
        db,
        req.query,
        vec,
        level=req.level,
        date_start=req.date_start,
        date_end=req.date_end,
        limit=req.limit,
    )
    return {"query": req.query, "mode": "hybrid", "results": [_serialize_node(r) for r in rows]}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def stats(request: fa.Request) -> dict[str, tp.Any]:
    db = _get_db(request)
    total = db.query_one("SELECT count() AS c FROM nodes")
    by_level = db.query(
        "SELECT level, count() AS n, min(date_start) AS d_min, max(date_end) AS d_max "
        "FROM nodes GROUP BY level ORDER BY level"
    )
    sources = db.query(
        "SELECT source, count() AS n FROM nodes GROUP BY source ORDER BY n DESC"
    )
    return {
        "total": int(total["c"]) if total else 0,
        "by_level": [
            {
                "level": r["level"],
                "count": int(r["n"]),
                "date_min": (
                    r["d_min"].isoformat()
                    if isinstance(r["d_min"], (dt.date, dt.datetime))
                    else r["d_min"]
                ),
                "date_max": (
                    r["d_max"].isoformat()
                    if isinstance(r["d_max"], (dt.date, dt.datetime))
                    else r["d_max"]
                ),
            }
            for r in by_level
        ],
        "sources": [{"source": r["source"], "count": int(r["n"])} for r in sources],
    }
