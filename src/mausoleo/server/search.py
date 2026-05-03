"""Search logic: semantic, text, hybrid (RRF)."""
from __future__ import annotations

import datetime as dt
import typing as tp

from mausoleo.server.db import Db


def _level_filter(level: str | None) -> tuple[str, dict[str, tp.Any]]:
    if not level:
        return "", {}
    return " AND level = {level:String}", {"level": level}


def _date_filter(
    date_start: str | None,
    date_end: str | None,
) -> tuple[str, dict[str, tp.Any]]:
    fragments: list[str] = []
    params: dict[str, tp.Any] = {}
    if date_start:
        fragments.append(" AND date_end >= {date_start:Date32}")
        params["date_start"] = dt.date.fromisoformat(date_start)
    if date_end:
        fragments.append(" AND date_start <= {date_end:Date32}")
        params["date_end"] = dt.date.fromisoformat(date_end)
    return "".join(fragments), params


def semantic_search(
    db: Db,
    query_embedding: list[float],
    *,
    level: str | None,
    date_start: str | None,
    date_end: str | None,
    limit: int,
) -> list[dict[str, tp.Any]]:
    if not query_embedding:
        return []
    lvl, lvl_params = _level_filter(level)
    dat, dat_params = _date_filter(date_start, date_end)
    sql = f"""
        SELECT
            node_id,
            level,
            parent_id,
            position,
            date_start,
            date_end,
            summary,
            child_count,
            L2Distance(embedding, {{q:Array(Float32)}}) AS distance
        FROM nodes
        WHERE 1{lvl}{dat}
        ORDER BY distance ASC
        LIMIT {{lim:UInt32}}
    """
    params: dict[str, tp.Any] = {"q": query_embedding, "lim": int(limit)}
    params.update(lvl_params)
    params.update(dat_params)
    return db.query(sql, params)


def text_search(
    db: Db,
    query: str,
    *,
    level: str | None,
    date_start: str | None,
    date_end: str | None,
    limit: int,
) -> list[dict[str, tp.Any]]:
    if not query.strip():
        return []
    lvl, lvl_params = _level_filter(level)
    dat, dat_params = _date_filter(date_start, date_end)
    # ``positionCaseInsensitive`` returns 0 when not found; we filter on >0.
    sql = f"""
        SELECT
            node_id,
            level,
            parent_id,
            position,
            date_start,
            date_end,
            summary,
            child_count,
            positionCaseInsensitive(summary, {{q:String}}) AS hit_pos
        FROM nodes
        WHERE positionCaseInsensitive(summary, {{q:String}}) > 0{lvl}{dat}
        ORDER BY level ASC, date_start ASC, position ASC
        LIMIT {{lim:UInt32}}
    """
    params: dict[str, tp.Any] = {"q": query, "lim": int(limit)}
    params.update(lvl_params)
    params.update(dat_params)
    return db.query(sql, params)


def hybrid_search(
    db: Db,
    query: str,
    query_embedding: list[float],
    *,
    level: str | None,
    date_start: str | None,
    date_end: str | None,
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, tp.Any]]:
    """Reciprocal Rank Fusion of semantic + text rankings.

    RRF score = sum over rankers of 1/(k + rank).
    """
    sem = semantic_search(
        db,
        query_embedding,
        level=level,
        date_start=date_start,
        date_end=date_end,
        limit=max(limit * 4, 40),
    )
    txt = text_search(
        db,
        query,
        level=level,
        date_start=date_start,
        date_end=date_end,
        limit=max(limit * 4, 40),
    )
    score: dict[str, float] = {}
    rec: dict[str, dict[str, tp.Any]] = {}
    for rank, row in enumerate(sem, start=1):
        score[row["node_id"]] = score.get(row["node_id"], 0.0) + 1.0 / (rrf_k + rank)
        rec[row["node_id"]] = row
    for rank, row in enumerate(txt, start=1):
        score[row["node_id"]] = score.get(row["node_id"], 0.0) + 1.0 / (rrf_k + rank)
        rec.setdefault(row["node_id"], row)
    out = sorted(rec.values(), key=lambda r: -score[r["node_id"]])[:limit]
    for r in out:
        r["rrf_score"] = score[r["node_id"]]
    return out
