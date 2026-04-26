"""Cross-page article completion via col1 blob mining.

The col1 sources (exp_105, exp_106, exp_113) often produce 5000-9000 char
blobs that contain MULTIPLE consecutive articles concatenated. The blob has
the page-N article body PLUS its page-N+1 continuation as one continuous
piece of text — but the matcher rejects it (Jaccard text_overlap < 0.30
because the blob is too long).

This post-processor:
  1. For each ensemble article on page N that ends mid-sentence (~truncated)
  2. Take its last K=80 chars (lower-cased, normalized)
  3. Search col1 blobs for fuzzy match of that tail
  4. If found, extract everything that follows up to the next likely article boundary
     (~600 chars or until we hit a multi-line break / capital-headline pattern)
  5. Append the extracted text as a new continuation paragraph + extend page_span

Usage:
    uv run --no-project python scripts/crosspage_completion.py <ens.json> <date> <out.json>
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


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _ends_unfinished(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?»\"')]}…"


def _find_fuzzy_position(haystack_norm: str, needle_norm: str, min_ratio: float = 0.85) -> int:
    if len(needle_norm) < 50:
        return -1
    exact = haystack_norm.find(needle_norm)
    if exact >= 0:
        return exact + len(needle_norm)
    matcher = difflib.SequenceMatcher(None, haystack_norm, needle_norm, autojunk=False)
    blocks = matcher.get_matching_blocks()
    best_idx = -1
    best_size = 0
    for b in blocks:
        if b.size > best_size and b.size >= len(needle_norm) * min_ratio:
            best_size = b.size
            best_idx = b.a + b.size
    return best_idx


def _find_segment_end(text: str, start: int, max_len: int = 800) -> int:
    end = min(len(text), start + max_len)
    region = text[start:end]
    cap_break = re.search(r"\n[A-ZÀÈÌÒÙ][A-ZÀÈÌÒÙ\s]{4,}", region)
    if cap_break:
        return start + cap_break.start()
    blank_break = region.rfind("\n\n")
    if blank_break > 100:
        return start + blank_break
    return end


def complete_crosspage(
    ensemble: dict[str, tp.Any],
    col1_predictions: list[dict[str, tp.Any]],
    tail_chars: int = 80,
    min_extension: int = 100,
) -> tuple[dict[str, tp.Any], int, int]:
    articles = list(ensemble.get("articles", []))
    blobs: list[str] = []
    for col1 in col1_predictions:
        for a in col1.get("articles", []):
            text = _article_text(a)
            if len(text) >= 2000:
                blobs.append(_normalize(text))

    if not blobs:
        return ensemble, 0, 0

    new_articles: list[dict[str, tp.Any]] = []
    completed = 0
    candidates = 0
    for art in articles:
        text = _article_text(art)
        if len(text) < 800 or not _ends_unfinished(text):
            new_articles.append(art)
            continue
        pages = art.get("page_span") or []
        if not pages or len(pages) > 1:
            new_articles.append(art)
            continue
        candidates += 1
        tail_raw = text[-tail_chars * 2:]
        tail = _normalize(tail_raw)[-tail_chars:]
        best_extension: str | None = None
        best_blob_idx = -1
        for bi, blob_norm in enumerate(blobs):
            seg_start = _find_fuzzy_position(blob_norm, tail)
            if seg_start < 0:
                continue
            seg_end = _find_segment_end(blob_norm, seg_start)
            extension = blob_norm[seg_start:seg_end].strip()
            if len(extension) >= min_extension:
                if best_extension is None or len(extension) > len(best_extension):
                    best_extension = extension
                    best_blob_idx = bi
        if best_extension is None:
            new_articles.append(art)
            continue
        new_art = dict(art)
        existing_paras = list(art.get("paragraphs", []))
        existing_paras.append({"text": best_extension, "is_continuation": True})
        new_art["paragraphs"] = existing_paras
        existing_pages = list(art.get("page_span", []))
        if existing_pages:
            extended_pages = sorted(set(existing_pages + [existing_pages[-1] + 1]))
            new_art["page_span"] = extended_pages
        new_articles.append(new_art)
        completed += 1

    return {**ensemble, "articles": new_articles}, candidates, completed


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: crosspage_completion.py <ens.json> <date> <out.json>")
        sys.exit(1)
    ens = json.loads(open(sys.argv[1]).read())
    date = sys.argv[2]
    out_path = sys.argv[3]
    import pathlib as pl

    cache = pl.Path("eval/predictions")
    col1_predictions: list[dict[str, tp.Any]] = []
    for src in COL1_SOURCES:
        path = cache / f"{src}_{date}.json"
        if path.exists():
            col1_predictions.append(json.loads(path.read_text()))

    out, candidates, completed = complete_crosspage(ens, col1_predictions)
    print(f"Truncated candidates: {candidates} | Completed: {completed}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
