from __future__ import annotations

import json
import re
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text


SENTENCE_END = (".", "!", "?", "»", ")", "]")
HYPHEN_ENDS = ("-", "—", "–")


def _text_of(art: dict[str, tp.Any]) -> str:
    return article_text(art)


def _ends_abruptly(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith(HYPHEN_ENDS):
        return True
    last = stripped[-1]
    if last in SENTENCE_END:
        return False
    if last.isalpha() or last == ",":
        return True
    return False


def _starts_continuation(art: dict[str, tp.Any]) -> bool:
    headline = art.get("headline")
    text = _text_of(art)
    if not text:
        return False
    if not headline or not str(headline).strip():
        return True
    first_word = re.match(r"[A-Za-zÀ-ÿ']+", text.strip())
    if first_word and first_word.group(0)[0].islower():
        return True
    if len(text.strip()) < 80 and not _ends_abruptly(text):
        return False
    return False


def _merge_article(first: dict[str, tp.Any], second: dict[str, tp.Any]) -> dict[str, tp.Any]:
    merged = dict(first)
    paras_a = list(first.get("paragraphs", []))
    paras_b = list(second.get("paragraphs", []))
    merged_paras = paras_a + paras_b
    merged["paragraphs"] = merged_paras
    ps_a = first.get("page_span") or []
    ps_b = second.get("page_span") or []
    all_pages = sorted(set((ps_a or []) + (ps_b or [])))
    if all_pages:
        merged["page_span"] = all_pages
    if not merged.get("headline") and second.get("headline"):
        merged["headline"] = second["headline"]
    return merged


def stitch_predictions(pred: dict[str, tp.Any]) -> dict[str, tp.Any]:
    articles = list(pred.get("articles", []))
    if not articles:
        return pred

    result: list[dict[str, tp.Any]] = []
    stitched = 0
    i = 0
    while i < len(articles):
        current = articles[i]
        current_text = _text_of(current)
        while i + 1 < len(articles):
            nxt = articles[i + 1]
            ps_cur = current.get("page_span") or []
            ps_nxt = nxt.get("page_span") or []
            same_or_adjacent = True
            if ps_cur and ps_nxt:
                max_cur = max(ps_cur)
                min_nxt = min(ps_nxt)
                same_or_adjacent = (min_nxt - max_cur) in (0, 1)
            if not same_or_adjacent:
                break
            if not _ends_abruptly(current_text):
                break
            if not _starts_continuation(nxt):
                break
            current = _merge_article(current, nxt)
            current_text = _text_of(current)
            stitched += 1
            i += 1
        result.append(current)
        i += 1

    out = dict(pred)
    out["articles"] = result
    print(f"  Stitched {stitched} continuations ({len(articles)} -> {len(result)})")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: stitch_articles.py <input> <output>")
        sys.exit(1)
    pred = json.loads(open(sys.argv[1]).read())
    stitched = stitch_predictions(pred)
    with open(sys.argv[2], "w") as f:
        json.dump(stitched, f, indent=2, ensure_ascii=False)
    print(f"Saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
