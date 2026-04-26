"""Split long col1/col2 VLM blobs using ensemble article boundaries as anchors.

For each prediction article > 3000 chars, search for the first ~60 chars of each
ensemble article inside it. When found, those mark article boundaries. Split
the blob into segments at those boundaries. Each segment becomes a new
prediction article (inheriting the matched ensemble article's headline).

Usage:
    uv run --no-project python scripts/blob_split.py <input.json> <output.json> <ensemble.json>
"""
from __future__ import annotations

import json
import re
import sys
import typing as tp


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def split_blobs(
    pred: dict[str, tp.Any],
    ensemble: dict[str, tp.Any],
    blob_threshold: int = 3000,
    anchor_len: int = 60,
    min_anchor_len: int = 30,
    min_segment: int = 50,
    min_separation: int = 50,
) -> dict[str, tp.Any]:
    anchors: list[tuple[str, dict[str, tp.Any]]] = []
    for a in ensemble.get("articles", []):
        text = _article_text(a)
        if not (100 <= len(text) <= 5000):
            continue
        norm = _normalize(text)[:anchor_len]
        if len(norm) >= min_anchor_len:
            anchors.append((norm, a))

    new_articles: list[dict[str, tp.Any]] = []
    for blob in pred.get("articles", []):
        text = _article_text(blob)
        if len(text) < blob_threshold:
            new_articles.append(blob)
            continue

        norm_text = _normalize(text)
        splits: list[tuple[int, dict[str, tp.Any]]] = []
        for anchor, src in anchors:
            idx = norm_text.find(anchor)
            if idx >= 0:
                splits.append((idx, src))
        splits.sort(key=lambda x: x[0])

        deduped: list[tuple[int, dict[str, tp.Any] | None]] = []
        last_idx = -min_separation - 1
        for idx, src in splits:
            if idx - last_idx > min_separation:
                deduped.append((idx, src))
                last_idx = idx

        if len(deduped) < 2:
            new_articles.append(blob)
            continue

        scale = len(text) / max(len(norm_text), 1)
        deduped.append((len(norm_text), None))
        for k in range(len(deduped) - 1):
            n_start, src = deduped[k]
            n_end = deduped[k + 1][0]
            t_start = int(n_start * scale)
            t_end = int(n_end * scale)
            segment = text[t_start:t_end].strip()
            if len(segment) >= min_segment:
                new_a = {
                    "unit_type": "article",
                    "headline": src.get("headline") if src else None,
                    "paragraphs": [{"text": segment, "is_continuation": False}],
                    "page_span": blob.get("page_span", []),
                }
                new_articles.append(new_a)

    return {**pred, "articles": new_articles}


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: blob_split.py <input.json> <output.json> <ensemble.json>")
        sys.exit(1)

    pred = json.loads(open(sys.argv[1]).read())
    ensemble = json.loads(open(sys.argv[3]).read())
    before = len(pred.get("articles", []))
    out = split_blobs(pred, ensemble)
    after = len(out.get("articles", []))
    print(f"Before: {before} | After: {after} | Added: {after - before}")
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
