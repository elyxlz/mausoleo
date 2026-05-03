"""Loader for the hierarchical index.

The loader walks two sources:

1. ``/tmp/mausoleo/eval/transcriptions/*.json`` — the OCR transcriptions (one
   file per day). Each contains articles with paragraphs. From these we derive
   level-0 (paragraph) and level-1 (article) nodes deterministically. When a
   matching summary file does not exist, we use the raw text as the summary.

2. ``/tmp/mausoleo/eval/summaries/{level}/*.json`` — per-level summary nodes
   produced by the parallel summarisation agent. Each file is a single JSON
   object representing a node with at least::

       {
           "node_id": "1943-07-01",
           "level": "day",
           "parent_id": "1943-07",
           "position": 0,
           "date_start": "1943-07-01",
           "date_end": "1943-07-01",
           "summary": "...",
           "embedding": [..],
           "child_count": 12
       }

   ``embedding`` may be missing — the loader fills with a zero vector of the
   configured dimensionality (default 1024 for BGE-M3) so the schema is
   satisfied and a real embedding pipeline can ``ALTER`` rows later.

Design constraints:

- Idempotent. Re-running the loader on the same data yields the same rows.
- Parent_ids are derived from the ID scheme described in
  ``plan/03_hierarchical_index.md`` so no second pass is needed.
- Tolerant of missing summary nodes: if no day/month/etc. summary file exists
  yet, we still emit a structural node with an empty summary so the tree is
  navigable end-to-end.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import logging
import pathlib as pl
import typing as tp

import clickhouse_connect

from mausoleo.index import schema
from mausoleo.index.models import Node

log = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 1024  # BGE-M3 dense
DEFAULT_TRANSCRIPTION_DIR = pl.Path("/tmp/mausoleo/eval/transcriptions")
DEFAULT_SUMMARY_DIR = pl.Path("/tmp/mausoleo/eval/summaries")


# ---------------------------------------------------------------------------
# ID derivation helpers
# ---------------------------------------------------------------------------

def date_for(node_id: str, level: str) -> tuple[dt.date, dt.date]:
    """Derive ``(date_start, date_end)`` from a deterministic node id."""
    if level == "archive":
        return dt.date(1880, 1, 1), dt.date(1945, 12, 31)
    if level == "decade":
        # e.g. "1940s"
        decade = int(node_id.rstrip("s"))
        return dt.date(decade, 1, 1), dt.date(decade + 9, 12, 31)
    if level == "year":
        y = int(node_id)
        return dt.date(y, 1, 1), dt.date(y, 12, 31)
    if level == "month":
        y, m = node_id.split("-")
        yi, mi = int(y), int(m)
        if mi == 12:
            end = dt.date(yi, 12, 31)
        else:
            end = dt.date(yi, mi + 1, 1) - dt.timedelta(days=1)
        return dt.date(yi, mi, 1), end
    if level == "day":
        d = dt.date.fromisoformat(node_id)
        return d, d
    if level == "article":
        # 1943-07-01_a000
        d = dt.date.fromisoformat(node_id.split("_")[0])
        return d, d
    if level == "paragraph":
        d = dt.date.fromisoformat(node_id.split("_")[0])
        return d, d
    raise ValueError(f"unknown level: {level}")


def parent_for(node_id: str, level: str) -> str:
    if level == "archive":
        return ""
    if level == "decade":
        return "archive"
    if level == "year":
        decade = (int(node_id) // 10) * 10
        return f"{decade}s"
    if level == "month":
        return node_id.split("-")[0]
    if level == "day":
        y, m, _ = node_id.split("-")
        return f"{y}-{m}"
    if level == "article":
        return node_id.split("_")[0]
    if level == "paragraph":
        return "_".join(node_id.split("_")[:2])
    raise ValueError(f"unknown level: {level}")


# ---------------------------------------------------------------------------
# Reading transcriptions and summaries
# ---------------------------------------------------------------------------

@dc.dataclass(frozen=True)
class LoaderConfig:
    transcription_dir: pl.Path = DEFAULT_TRANSCRIPTION_DIR
    summary_dir: pl.Path = DEFAULT_SUMMARY_DIR
    embed_dim: int = DEFAULT_EMBED_DIM
    source: str = "il_messaggero"
    # Date filter (inclusive). When set, only days whose date falls in this
    # window become part of the corpus and only the months/year/decade/archive
    # spanning that window are emitted.
    date_start: dt.date | None = None
    date_end: dt.date | None = None


def _zeros(dim: int) -> list[float]:
    return [0.0] * dim


def _load_summary_index(cfg: LoaderConfig) -> dict[str, dict[str, tp.Any]]:
    """Load any summary JSONs the parallel agent has produced.

    Returns a flat ``node_id -> payload`` mapping. Missing — fine, returns
    an empty dict.
    """
    index: dict[str, dict[str, tp.Any]] = {}
    if not cfg.summary_dir.exists():
        return index
    for level_dir in cfg.summary_dir.iterdir():
        if not level_dir.is_dir():
            continue
        for f in level_dir.glob("*.json"):
            try:
                payload = json.loads(f.read_text())
            except Exception as exc:  # pragma: no cover - corrupt json
                log.warning("failed to parse %s: %s", f, exc)
                continue
            nid = payload.get("node_id") or f.stem
            index[nid] = payload
    return index


def _enrich(
    node_id: str,
    level: str,
    position: int,
    summaries: dict[str, dict[str, tp.Any]],
    cfg: LoaderConfig,
    *,
    fallback_summary: str = "",
    raw_text: str | None = None,
) -> Node:
    payload = summaries.get(node_id, {})
    summary = payload.get("summary") or fallback_summary
    embedding = payload.get("embedding") or _zeros(cfg.embed_dim)
    child_count = int(payload.get("child_count", 0))
    parent_id = payload.get("parent_id") or parent_for(node_id, level)
    if "date_start" in payload and "date_end" in payload:
        ds = dt.date.fromisoformat(payload["date_start"])
        de = dt.date.fromisoformat(payload["date_end"])
    else:
        ds, de = date_for(node_id, level)
    return Node(
        node_id=node_id,
        level=tp.cast(tp.Any, level),
        parent_id=parent_id,
        position=int(payload.get("position", position)),
        date_start=ds,
        date_end=de,
        source=payload.get("source", cfg.source),
        summary=summary,
        raw_text=raw_text,
        embedding=list(embedding),
        child_count=child_count,
    )


def _in_range(d: dt.date, cfg: LoaderConfig) -> bool:
    if cfg.date_start is not None and d < cfg.date_start:
        return False
    if cfg.date_end is not None and d > cfg.date_end:
        return False
    return True


def build_nodes(cfg: LoaderConfig | None = None) -> list[Node]:
    """Walk the input directories and produce a list of ``Node`` objects."""
    cfg = cfg or LoaderConfig()
    summaries = _load_summary_index(cfg)
    out: list[Node] = []

    # ---- Level 0/1 from transcriptions --------------------------------
    if cfg.transcription_dir.exists():
        days_seen: set[str] = set()
        months_seen: set[str] = set()
        years_seen: set[str] = set()
        decades_seen: set[str] = set()

        for f in sorted(cfg.transcription_dir.glob("*.json")):
            try:
                doc = json.loads(f.read_text())
            except Exception as exc:  # pragma: no cover
                log.warning("bad transcription %s: %s", f, exc)
                continue
            try:
                day_date = dt.date.fromisoformat(doc["date"])
            except Exception:
                log.warning("missing/invalid date in %s", f)
                continue
            if not _in_range(day_date, cfg):
                continue

            day_id = doc["date"]
            articles = doc.get("articles", [])
            day_child_count = len(articles)

            # paragraphs + articles
            for art in articles:
                art_id = art["id"]
                paragraphs = art.get("paragraphs", [])
                for p in paragraphs:
                    pid = p["id"]
                    txt = p.get("text", "")
                    pos = int(pid.rsplit("_p", 1)[-1])
                    out.append(
                        _enrich(
                            pid,
                            "paragraph",
                            pos,
                            summaries,
                            cfg,
                            fallback_summary=txt[:400],
                            raw_text=txt,
                        )
                    )
                pos = int(art.get("position_in_issue", art_id.rsplit("_a", 1)[-1]))
                headline = art.get("headline", "")
                fallback = headline if headline else (
                    paragraphs[0].get("text", "")[:400] if paragraphs else ""
                )
                node = _enrich(
                    art_id,
                    "article",
                    pos,
                    summaries,
                    cfg,
                    fallback_summary=fallback,
                )
                node = dc.replace(node, child_count=len(paragraphs))
                out.append(node)

            # day node
            day_node = _enrich(
                day_id,
                "day",
                day_date.toordinal(),
                summaries,
                cfg,
                fallback_summary=f"{day_id}: {day_child_count} articles",
            )
            day_node = dc.replace(day_node, child_count=day_child_count)
            out.append(day_node)
            days_seen.add(day_id)
            months_seen.add(day_id[:7])
            years_seen.add(day_id[:4])
            decades_seen.add(f"{(int(day_id[:4]) // 10) * 10}s")

        # ---- months / years / decades / archive ---------------------------
        for mid in sorted(months_seen):
            n_days = sum(1 for d in days_seen if d.startswith(mid))
            yi, mi = mid.split("-")
            pos = int(yi) * 12 + int(mi)
            mn = _enrich(
                mid,
                "month",
                pos,
                summaries,
                cfg,
                fallback_summary=f"{mid}: {n_days} days indexed",
            )
            mn = dc.replace(mn, child_count=n_days)
            out.append(mn)

        for yid in sorted(years_seen):
            n_months = sum(1 for m in months_seen if m.startswith(yid))
            yn = _enrich(
                yid,
                "year",
                int(yid),
                summaries,
                cfg,
                fallback_summary=f"{yid}: {n_months} months indexed",
            )
            yn = dc.replace(yn, child_count=n_months)
            out.append(yn)

        for did in sorted(decades_seen):
            n_years = sum(1 for y in years_seen if (int(y) // 10) * 10 == int(did.rstrip("s")))
            dn = _enrich(
                did,
                "decade",
                int(did.rstrip("s")),
                summaries,
                cfg,
                fallback_summary=f"{did}: {n_years} years indexed",
            )
            dn = dc.replace(dn, child_count=n_years)
            out.append(dn)

        if decades_seen:
            an = _enrich(
                "archive",
                "archive",
                0,
                summaries,
                cfg,
                fallback_summary="il_messaggero archive: full corpus",
            )
            an = dc.replace(an, child_count=len(decades_seen))
            out.append(an)

    # ---- Any extra nodes the summary agent emitted ourselves wouldn't have
    # generated above (e.g. a custom 'archive' summary the agent wrote on its
    # own without raw transcriptions) — include them too.
    seen_ids = {n.node_id for n in out}
    for nid, payload in summaries.items():
        if nid in seen_ids:
            continue
        level = payload.get("level")
        if not level:
            continue
        try:
            ds, de = (
                dt.date.fromisoformat(payload["date_start"]),
                dt.date.fromisoformat(payload["date_end"]),
            )
        except Exception:
            ds, de = date_for(nid, level)
        out.append(
            Node(
                node_id=nid,
                level=tp.cast(tp.Any, level),
                parent_id=payload.get("parent_id") or parent_for(nid, level),
                position=int(payload.get("position", 0)),
                date_start=ds,
                date_end=de,
                source=payload.get("source", cfg.source),
                summary=payload.get("summary", ""),
                raw_text=payload.get("raw_text"),
                embedding=list(payload.get("embedding") or _zeros(cfg.embed_dim)),
                child_count=int(payload.get("child_count", 0)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# ClickHouse writes
# ---------------------------------------------------------------------------

def _client(host: str, port: int, database: str) -> tp.Any:
    return clickhouse_connect.get_client(host=host, port=port, database=database)


def setup_schema(host: str = "127.0.0.1", port: int = 8123, database: str = "default") -> None:
    """Create the ``nodes`` table and indexes if missing."""
    cli = _client(host, port, database)
    for stmt in schema.all_setup_statements():
        try:
            cli.command(stmt)
        except Exception as exc:
            # vector_similarity index requires an experimental setting; ignore
            # if it can't be created — semantic search still works via brute
            # force L2Distance.
            log.warning("schema stmt failed (%s): %s", stmt[:60], exc)


def _to_clickhouse_rows(nodes: list[Node]) -> tuple[list[str], list[list[tp.Any]]]:
    cols = [
        "node_id",
        "level",
        "parent_id",
        "position",
        "date_start",
        "date_end",
        "source",
        "summary",
        "raw_text",
        "embedding",
        "child_count",
    ]
    rows = [
        [
            n.node_id,
            n.level,
            n.parent_id,
            int(n.position),
            n.date_start,
            n.date_end,
            n.source,
            n.summary,
            n.raw_text,
            n.embedding,
            int(n.child_count),
        ]
        for n in nodes
    ]
    return cols, rows


def insert_nodes(
    nodes: list[Node],
    *,
    host: str = "127.0.0.1",
    port: int = 8123,
    database: str = "default",
    batch_size: int = 500,
    truncate: bool = False,
) -> int:
    cli = _client(host, port, database)
    if truncate:
        cli.command("TRUNCATE TABLE IF EXISTS nodes")
    cols, rows = _to_clickhouse_rows(nodes)
    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        cli.insert("nodes", batch, column_names=cols)
        inserted += len(batch)
    return inserted


def load(
    cfg: LoaderConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8123,
    database: str = "default",
    truncate: bool = True,
) -> dict[str, tp.Any]:
    cfg = cfg or LoaderConfig()
    setup_schema(host=host, port=port, database=database)
    nodes = build_nodes(cfg)
    n = insert_nodes(
        nodes,
        host=host,
        port=port,
        database=database,
        truncate=truncate,
    )
    by_level: dict[str, int] = {}
    for nd in nodes:
        by_level[nd.level] = by_level.get(nd.level, 0) + 1
    return {"inserted": n, "by_level": by_level}
