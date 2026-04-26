"""Replace truncated ensemble articles with col1 article + its continuation.

Algorithm:
  1. For each col1 source, scan its articles in order. When article[i] has a
     headline AND article[i+1] has no headline (or very short) AND starts
     lowercase, treat them as a (head_part, continuation) pair.
  2. For each ensemble article that ends mid-sentence, find a (head_part,
     continuation) pair in any col1 source where head_part starts the same way.
  3. Replace the ensemble article's text with head_part + " " + continuation
     and extend page_span by 1.

Usage:
    uv run --no-project python scripts/blob_replace_v2.py <ens.json> <date> <out.json>
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


def _ends_unfinished(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?»\"')]}…"


def _starts_match(a: str, b: str, min_chars: int = 60, min_ratio: float = 0.85) -> bool:
    a_start = _normalize(a)[:min_chars]
    b_start = _normalize(b)[:min_chars]
    if len(a_start) < min_chars or len(b_start) < min_chars:
        return False
    return difflib.SequenceMatcher(None, a_start, b_start, autojunk=False).ratio() >= min_ratio


def _is_continuation_article(article: dict[str, tp.Any]) -> bool:
    text = _article_text(article).lstrip()
    if not text:
        return False
    headline = article.get("headline") or ""
    if len(headline) > 20:
        return False
    first = text[0]
    if not first.isalpha():
        return False
    if not first.islower():
        return False
    return True


def collect_pairs(col1: dict[str, tp.Any], max_ahead: int = 12) -> list[tuple[str, str]]:
    arts = col1.get("articles", [])
    pairs: list[tuple[str, str]] = []
    for i in range(len(arts) - 1):
        head_text = _article_text(arts[i])
        if len(head_text) < 300:
            continue
        if not _ends_unfinished(head_text):
            continue
        for j in range(i + 1, min(i + 1 + max_ahead, len(arts))):
            cont = arts[j]
            if not _is_continuation_article(cont):
                continue
            cont_text = _article_text(cont)
            if len(cont_text) < 200:
                continue
            pairs.append((head_text, cont_text))
            break
    return pairs


def replace_with_pairs(
    ensemble: dict[str, tp.Any],
    col1_predictions: list[dict[str, tp.Any]],
) -> tuple[dict[str, tp.Any], int, int]:
    pairs: list[tuple[str, str]] = []
    for col1 in col1_predictions:
        pairs.extend(collect_pairs(col1))

    articles = list(ensemble.get("articles", []))
    new_articles = list(articles)
    replaced = 0
    for ai, art in enumerate(articles):
        text = _article_text(art)
        if len(text) < 400:
            continue
        if not _ends_unfinished(text):
            continue
        original_pages = art.get("page_span") or []
        if len(original_pages) > 1:
            continue
        best_combined: str | None = None
        best_total_len = 0
        for head_text, cont_text in pairs:
            if not _starts_match(text, head_text):
                continue
            combined = head_text.rstrip() + " " + cont_text.lstrip()
            if len(combined) < len(text) * 1.15:
                continue
            if len(combined) > best_total_len:
                best_total_len = len(combined)
                best_combined = combined
        if best_combined is None:
            continue
        new_art = dict(art)
        new_art["paragraphs"] = [{"text": best_combined, "is_continuation": False}]
        if original_pages:
            new_art["page_span"] = sorted(set(original_pages + [original_pages[-1] + 1]))
        new_articles[ai] = new_art
        replaced += 1

    return {**ensemble, "articles": new_articles}, len(pairs), replaced


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: blob_replace_v2.py <ens.json> <date> <out.json>")
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

    out, pairs_seen, replaced = replace_with_pairs(ens, col1_predictions)
    print(f"Continuation pairs found: {pairs_seen} | Articles replaced: {replaced}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
