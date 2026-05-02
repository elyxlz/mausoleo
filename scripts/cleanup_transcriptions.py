"""Cleanup ensemble OCR predictions into final transcriptions.

Reads eval/predictions/ensemble_30min_<date>.json, applies:
  - dedup near-duplicate articles (same/similar headline + overlapping text)
  - drop garbage (very short, page headers, JSON blobs)
  - stitch fragments (headline-only article followed by no-headline body)
  - cross-page stitching (article ending mid-sentence + next no-headline starting mid-sentence)

Writes eval/transcriptions/<date>.json with cleaned articles.
"""
from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
import re
import sys
import typing as tp
from difflib import SequenceMatcher


PRED_DIR = pl.Path("eval/predictions")
OUT_DIR = pl.Path("eval/transcriptions")
PRED_GLOB = "ensemble_30min_*.json"


def normalize_alnum(s: str | None) -> str:
    return "".join(c.lower() for c in (s or "") if c.isalnum())


def article_text(art: dict[str, tp.Any]) -> str:
    return "\n\n".join(p.get("text", "") for p in art.get("paragraphs", []))


def article_chars(art: dict[str, tp.Any]) -> int:
    return sum(len(p.get("text", "")) for p in art.get("paragraphs", []))


_PAGE_HEADER_RE = re.compile(
    r"^\s*IL MESSAGGERO\s*[-–—]\s*\w+\s+\d+\s+\w+\s+19\d{2}", re.IGNORECASE
)
_TRUNC_END_RE = re.compile(r"[a-z,;\-—–](\s|$)$")
_LOWER_START_RE = re.compile(r"^[a-zàèéìòù]")


def is_page_header(art: dict[str, tp.Any]) -> bool:
    body = article_text(art).strip()
    if _PAGE_HEADER_RE.match(body):
        return True
    if len(body) < 80 and "MESSAGGERO" in body.upper() and re.search(r"19\d{2}", body):
        return True
    return False


_HEADLINE_TRUNCATED_RE = re.compile(r"^[a-zàèéìòù]|^[A-Z][a-z]*[A-Z]")


def is_truncated_headline(h: str) -> bool:
    h = h.strip()
    if not h:
        return False
    first_word = h.split()[0] if h.split() else ""
    if not first_word:
        return False
    if first_word[0].islower() and not first_word[0] in "lda":
        return True
    if first_word in {"di", "el", "al", "il", "la", "le", "lo", "del", "dal"}:
        return False
    truncated_starts = (
        "ACCIA", "ATTAGLIA", "EL MEDITERRANEO", "RICANI", "OGGIO", "ortofrutticoli",
        "AN SOCIE", "OGGI al SU", "GI ", "incrociatore",
    )
    if any(h.startswith(s) for s in truncated_starts):
        return True
    return False


def is_garbage(art: dict[str, tp.Any]) -> bool:
    body = article_text(art).strip()
    headline = (art.get("headline") or "").strip()
    if is_page_header(art):
        return True
    if not body and len(headline) < 8:
        return True
    if len(body) < 20 and not headline:
        return True
    if body.startswith(("{", "```", '"articles"')) and len(body) < 500:
        return True
    if body and len(body.replace(" ", "").replace("\n", "")) < 5:
        return True
    if len(body) < 100 and is_truncated_headline(headline):
        return True
    if len(body) < 150 and not headline and not body[:1].isupper():
        return True
    return False


def text_similarity(a: str, b: str, sample: int = 800) -> float:
    na = normalize_alnum(a)[:sample]
    nb = normalize_alnum(b)[:sample]
    if not na or not nb:
        return 0.0
    if abs(len(na) - len(nb)) / max(len(na), len(nb)) > 0.7:
        return 0.0
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def headline_bucket(art: dict[str, tp.Any]) -> str:
    h = normalize_alnum(art.get("headline"))
    return h[:30]


def pick_best(arts: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
    return max(arts, key=lambda a: (article_chars(a), len(a.get("headline") or "")))


def _trigram_set(text: str, n: int = 6) -> frozenset[str]:
    nt = normalize_alnum(text)
    return frozenset(nt[i : i + n] for i in range(0, max(0, len(nt) - n + 1), 3))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_articles(articles: list[dict[str, tp.Any]]) -> tuple[list[dict[str, tp.Any]], int]:
    drop: set[int] = set()

    buckets: dict[str, list[int]] = {}
    for i, a in enumerate(articles):
        key = headline_bucket(a)
        if not key:
            continue
        buckets.setdefault(key, []).append(i)

    for key, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        groups: list[list[int]] = []
        for i in idxs:
            if i in drop:
                continue
            placed = False
            for g in groups:
                if text_similarity(article_text(articles[g[0]]), article_text(articles[i])) > 0.55:
                    g.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
        for g in groups:
            if len(g) < 2:
                continue
            best = max(g, key=lambda j: (article_chars(articles[j]), len(articles[j].get("headline") or "")))
            for j in g:
                if j != best:
                    drop.add(j)

    cands = [(i, a, article_chars(a)) for i, a in enumerate(articles) if i not in drop and article_chars(a) >= 80]
    trigrams: dict[int, frozenset[str]] = {i: _trigram_set(article_text(a)) for i, a, _ in cands}

    cands.sort(key=lambda t: t[2])
    for ai in range(len(cands)):
        i, a, ca = cands[ai]
        if i in drop:
            continue
        ta = trigrams[i]
        if len(ta) < 12:
            continue
        for bi in range(ai + 1, len(cands)):
            j, b, cb = cands[bi]
            if j in drop:
                continue
            tb = trigrams[j]
            jac = _jaccard(ta, tb)
            if jac < 0.35:
                continue
            sim = text_similarity(article_text(a), article_text(b), sample=2000)
            containment = (len(ta & tb) / len(ta)) if ta else 0.0
            if sim > 0.55 or containment > 0.7:
                if cb >= ca:
                    drop.add(i)
                    break
                else:
                    drop.add(j)

    kept = [a for i, a in enumerate(articles) if i not in drop]
    return kept, len(drop)


def stitch_fragments(articles: list[dict[str, tp.Any]]) -> tuple[list[dict[str, tp.Any]], int]:
    out: list[dict[str, tp.Any]] = []
    stitched = 0
    skip_next = 0
    for i, a in enumerate(articles):
        if skip_next > 0:
            skip_next -= 1
            continue
        body = article_text(a).strip()
        headline = (a.get("headline") or "").strip()
        if (
            i + 1 < len(articles)
            and headline
            and len(body) < 100
            and a.get("unit_type") == "article"
        ):
            nxt = articles[i + 1]
            nxt_body = article_text(nxt).strip()
            if (
                not (nxt.get("headline") or "").strip()
                and len(nxt_body) > 200
                and nxt.get("unit_type") == "article"
            ):
                merged = dict(a)
                merged_paragraphs = list(a.get("paragraphs", []))
                merged_paragraphs.extend(nxt.get("paragraphs", []))
                merged["paragraphs"] = merged_paragraphs
                merged["page_span"] = sorted(set(a.get("page_span", []) + nxt.get("page_span", [])))
                out.append(merged)
                skip_next = 1
                stitched += 1
                continue
        out.append(a)
    return out, stitched


def stitch_crosspage(articles: list[dict[str, tp.Any]]) -> tuple[list[dict[str, tp.Any]], int]:
    out: list[dict[str, tp.Any]] = []
    stitched = 0
    skip_next = 0
    for i, a in enumerate(articles):
        if skip_next > 0:
            skip_next -= 1
            continue
        body = article_text(a).rstrip()
        if (
            i + 1 < len(articles)
            and a.get("unit_type") == "article"
            and body
            and _TRUNC_END_RE.search(body)
            and len(body) > 150
        ):
            nxt = articles[i + 1]
            nxt_body = article_text(nxt).lstrip()
            if (
                not (nxt.get("headline") or "").strip()
                and nxt.get("unit_type") == "article"
                and nxt_body
                and (_LOWER_START_RE.match(nxt_body) or len(nxt_body) > 80)
            ):
                merged = dict(a)
                merged_paragraphs = list(a.get("paragraphs", []))
                merged_paragraphs.extend(nxt.get("paragraphs", []))
                merged["paragraphs"] = merged_paragraphs
                merged["page_span"] = sorted(set(a.get("page_span", []) + nxt.get("page_span", [])))
                out.append(merged)
                skip_next = 1
                stitched += 1
                continue
        out.append(a)
    return out, stitched


def filter_garbage(articles: list[dict[str, tp.Any]]) -> tuple[list[dict[str, tp.Any]], int]:
    kept = []
    dropped = 0
    for a in articles:
        if is_garbage(a):
            dropped += 1
        else:
            kept.append(a)
    return kept, dropped


def reindex(articles: list[dict[str, tp.Any]], date: str) -> list[dict[str, tp.Any]]:
    out = []
    for i, a in enumerate(articles):
        new_a = dict(a)
        new_a["id"] = f"{date}_a{i:03d}"
        new_a["position_in_issue"] = i
        new_paras = []
        for j, p in enumerate(a.get("paragraphs", [])):
            np = dict(p)
            np["id"] = f"{date}_a{i:03d}_p{j:02d}"
            new_paras.append(np)
        new_a["paragraphs"] = new_paras
        out.append(new_a)
    return out


def cleanup_issue(pred: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], dict[str, int]]:
    arts = pred["articles"]
    n0 = len(arts)
    arts, n_garbage = filter_garbage(arts)
    arts, n_stitch_frag = stitch_fragments(arts)
    arts, n_stitch_xp = stitch_crosspage(arts)
    arts, n_dup = dedup_articles(arts)
    arts = reindex(arts, pred["date"])
    cleaned = {
        "date": pred["date"],
        "source": pred["source"],
        "page_count": pred["page_count"],
        "articles": arts,
    }
    stats = {
        "before": n0,
        "after": len(arts),
        "dropped_garbage": n_garbage,
        "stitched_fragments": n_stitch_frag,
        "stitched_crosspage": n_stitch_xp,
        "deduped": n_dup,
    }
    return cleaned, stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PRED_DIR.glob(PRED_GLOB))
    if not paths:
        print("no predictions found")
        return
    total_stats = {"before": 0, "after": 0, "dropped_garbage": 0, "stitched_fragments": 0, "stitched_crosspage": 0, "deduped": 0}
    print(f"{'date':<12} {'before':>6} {'after':>6} {'-garb':>6} {'stitch_f':>8} {'stitch_x':>8} {'-dup':>6}")
    for p in paths:
        pred = json.loads(p.read_text())
        cleaned, stats = cleanup_issue(pred)
        out_path = OUT_DIR / f"{cleaned['date']}.json"
        out_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
        for k, v in stats.items():
            total_stats[k] += v
        print(f"{cleaned['date']:<12} {stats['before']:>6} {stats['after']:>6} {stats['dropped_garbage']:>6} {stats['stitched_fragments']:>8} {stats['stitched_crosspage']:>8} {stats['deduped']:>6}")
    print(f"\n{'TOTAL':<12} {total_stats['before']:>6} {total_stats['after']:>6} {total_stats['dropped_garbage']:>6} {total_stats['stitched_fragments']:>8} {total_stats['stitched_crosspage']:>8} {total_stats['deduped']:>6}")


if __name__ == "__main__":
    main()
