from __future__ import annotations

import json
import statistics as st
import typing as tp

from mausoleo.eval.evaluate import article_text
from mausoleo.paths import EVAL_DATES, PRED_DIR

WEAK_PAGE_RATIO = 0.35
BLOB_RATIO = 3.0
TINY_CHARS = 50


def page_totals(articles: list[dict[str, tp.Any]]) -> dict[int, tuple[int, int]]:
    totals: dict[int, tuple[int, int]] = {}
    for article in articles:
        page = (article.get("page_span") or [1])[0]
        count, chars = totals.get(page, (0, 0))
        totals[page] = (count + 1, chars + len(article_text(article)))
    return totals


def weak_pages(articles: list[dict[str, tp.Any]], page_count: int) -> list[dict[str, tp.Any]]:
    totals = page_totals(articles)
    present = [totals.get(p, (0, 0))[1] for p in range(1, page_count + 1)]
    if not present:
        return []
    median = st.median([c for c in present if c > 0] or [0])
    weak = []
    for page in range(1, page_count + 1):
        count, chars = totals.get(page, (0, 0))
        if median > 0 and chars < median * WEAK_PAGE_RATIO:
            weak.append({"page": page, "articles": count, "chars": chars, "median_page_chars": int(median), "ratio": round(chars / median, 3)})
    return weak


def issue_diagnostics(pred: dict[str, tp.Any]) -> dict[str, tp.Any]:
    articles = pred.get("articles", [])
    lengths = [len(article_text(a)) for a in articles]
    page_count = pred.get("page_count") or (max((a.get("page_span") or [1])[0] for a in articles) if articles else 0)
    return {
        "articles": len(articles),
        "chars": sum(lengths),
        "pages_with_content": len(page_totals(articles)),
        "page_count": page_count,
        "weak_pages": weak_pages(articles, page_count),
        "empty_pages": [p for p in range(1, page_count + 1) if p not in page_totals(articles)],
        "tiny_articles": sum(1 for n in lengths if n < TINY_CHARS),
        "single_paragraph": sum(1 for a in articles if len(a.get("paragraphs", [])) == 1),
        "no_headline": sum(1 for a in articles if not a.get("headline")),
        "median_chars": int(st.median(lengths)) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
    }


def config_diagnostics(config: str, dates: tp.Sequence[str] = EVAL_DATES) -> dict[str, dict[str, tp.Any]]:
    out: dict[str, dict[str, tp.Any]] = {}
    for date in dates:
        path = PRED_DIR / f"{config}_{date}.json"
        if not path.exists():
            continue
        out[date] = issue_diagnostics(json.loads(path.read_text()))
    return out


def report_lines(config: str, dates: tp.Sequence[str] = EVAL_DATES) -> list[str]:
    diags = config_diagnostics(config, dates)
    lines = [f"{'issue':<12}{'arts':>6}{'chars':>9}{'pages':>7}{'median':>8}{'tiny':>6}{'1-para':>8}{'no-hl':>7}  flags"]
    for date, d in diags.items():
        flags = []
        if d["empty_pages"]:
            flags.append(f"EMPTY pages {d['empty_pages']}")
        for w in d["weak_pages"]:
            flags.append(f"WEAK page {w['page']} ({w['chars']}c = {w['ratio']:.0%} of median, {w['articles']} arts)")
        lines.append(
            f"{date:<12}{d['articles']:>6}{d['chars']:>9,}{d['pages_with_content']:>4}/{d['page_count']:<2}"
            f"{d['median_chars']:>8}{d['tiny_articles']:>6}{d['single_paragraph']:>8}{d['no_headline']:>7}  "
            + ("; ".join(flags) if flags else "ok")
        )
    return lines
