"""Replace truncated ensemble articles with longer text mined from col1 blobs.

Algorithm:
  1. For each col1 blob (>2000 chars), find positions of every ensemble article's
     first 60 chars within the blob (these are the article boundaries).
  2. Split the blob at those positions; each split segment represents the col1
     model's complete view of one article (including any cross-page continuation).
  3. For each split segment, locate the matching ensemble article by its anchor.
  4. If the split segment is at least 1.3x longer than the ensemble article AND
     the first 50 chars of both match closely, REPLACE the ensemble article's
     paragraphs with the segment text.
  5. Extend the article's page_span by 1 (since col1 was page-aware).

Usage:
    uv run --no-project python scripts/blob_replace.py <ens.json> <date> <out.json>
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import typing as tp


COL1_SOURCES = [
    "exp_105_col1_qwen3vl_vllm",
    "exp_106_col1_ads_vllm",
    "exp_113_col1_qwen25vl",
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _starts_match(a: str, b: str, min_chars: int = 50, min_ratio: float = 0.85) -> bool:
    a_start = _normalize(a)[:min_chars]
    b_start = _normalize(b)[:min_chars]
    if len(a_start) < min_chars or len(b_start) < min_chars:
        return False
    ratio = difflib.SequenceMatcher(None, a_start, b_start, autojunk=False).ratio()
    return ratio >= min_ratio


def replace_with_blob_segments(
    ensemble: dict[str, tp.Any],
    col1_predictions: list[dict[str, tp.Any]],
    anchor_len: int = 60,
    min_anchor: int = 30,
    length_threshold: float = 1.2,
) -> tuple[dict[str, tp.Any], int, int]:
    articles = list(ensemble.get("articles", []))
    anchors: list[tuple[str, int]] = []
    for ai, art in enumerate(articles):
        text = _article_text(art)
        if not (200 <= len(text) <= 5000):
            continue
        norm = _normalize(text)[:anchor_len]
        if len(norm) >= min_anchor:
            anchors.append((norm, ai))

    candidates: dict[int, list[str]] = {}
    blobs_seen = 0
    for col1 in col1_predictions:
        for blob in col1.get("articles", []):
            blob_text = _article_text(blob)
            if len(blob_text) < 2000:
                continue
            blobs_seen += 1
            blob_norm = _normalize(blob_text)
            blob_scale = len(blob_text) / max(len(blob_norm), 1)
            hits: list[tuple[int, int]] = []
            for anchor, ai in anchors:
                pos = blob_norm.find(anchor)
                if pos >= 0:
                    hits.append((pos, ai))
            hits.sort(key=lambda x: x[0])

            if len(hits) < 1:
                continue

            for k, (pos, ai) in enumerate(hits):
                next_pos = hits[k + 1][0] if k + 1 < len(hits) else len(blob_norm)
                t_start = int(pos * blob_scale)
                t_end = int(next_pos * blob_scale)
                segment = blob_text[t_start:t_end].strip()
                if len(segment) < 100:
                    continue
                candidates.setdefault(ai, []).append(segment)

    new_articles: list[dict[str, tp.Any]] = list(articles)
    replaced = 0
    max_extension_factor = 1.6
    for ai, segments in candidates.items():
        original = articles[ai]
        original_text = _article_text(original)
        if not _ends_unfinished(original_text):
            continue
        original_len = len(original_text)
        if original_len < 600:
            continue
        original_pages = original.get("page_span") or []
        if len(original_pages) > 1:
            continue
        max_len = max(int(original_len * max_extension_factor), original_len + 600)
        best_segment: str | None = None
        for seg in segments:
            if not _starts_match(original_text, seg):
                continue
            if len(seg) < original_len * length_threshold:
                continue
            truncated = seg[:max_len].rstrip()
            if best_segment is None or len(truncated) > len(best_segment):
                best_segment = truncated
        if best_segment is None:
            continue
        new_art = dict(original)
        new_art["paragraphs"] = [{"text": best_segment, "is_continuation": False}]
        if original_pages:
            new_art["page_span"] = sorted(set(original_pages + [original_pages[-1] + 1]))
        new_articles[ai] = new_art
        replaced += 1

    return {**ensemble, "articles": new_articles}, blobs_seen, replaced


def _ends_unfinished(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?»\"')]}…"


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: blob_replace.py <ens.json> <date> <out.json>")
        sys.exit(1)
    import pathlib as pl

    ens = json.loads(open(sys.argv[1]).read())
    date = sys.argv[2]
    out_path = sys.argv[3]
    cache = pl.Path("eval/predictions")
    col1_predictions: list[dict[str, tp.Any]] = []
    for src in COL1_SOURCES:
        path = cache / f"{src}_{date}.json"
        if path.exists():
            col1_predictions.append(json.loads(path.read_text()))

    out, blobs_seen, replaced = replace_with_blob_segments(ens, col1_predictions)
    print(f"Blobs scanned: {blobs_seen} | Articles replaced: {replaced}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
