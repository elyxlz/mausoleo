from __future__ import annotations

import dataclasses as dc
import json
import re
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BOLD_LINE_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
_EMPHASIS_RE = re.compile(r"(\*\*|__|(?<!\w)[*_](?!\s))")
_IMAGE_TAG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>|<!--.*?-->", re.DOTALL)


def _clean_inline(text: str) -> str:
    text = _IMAGE_TAG_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    return text.strip()


def _heading_text(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    if m:
        return _clean_inline(m.group(1))
    m = _BOLD_LINE_RE.match(line)
    if m:
        return _clean_inline(m.group(1))
    return None


def _build_article(headline: str | None, lines: list[str]) -> dict[str, tp.Any] | None:
    body = "\n".join(lines).strip()
    paragraphs = [{"text": _clean_inline(p)} for p in re.split(r"\n\s*\n", body) if _clean_inline(p)]
    if not paragraphs and not headline:
        return None
    if not paragraphs:
        paragraphs = [{"text": headline or ""}]
    return {"unit_type": "article", "headline": headline, "paragraphs": paragraphs}


def split_markdown_articles(markdown: str) -> list[dict[str, tp.Any]]:
    articles: list[dict[str, tp.Any]] = []
    headline: str | None = None
    lines: list[str] = []
    for line in markdown.splitlines():
        new_headline = _heading_text(line)
        if new_headline is None:
            lines.append(line)
            continue
        article = _build_article(headline, lines)
        if article:
            articles.append(article)
        headline, lines = new_headline, []
    article = _build_article(headline, lines)
    if article:
        articles.append(article)
    return articles


@dc.dataclass(frozen=True, kw_only=True)
class MergeMarkdownPages(BaseOperatorConfig):
    pass


@register_operator(MergeMarkdownPages, operation=OperatorType.MAP)
def merge_markdown_pages(row: dict[str, tp.Any], *, config: MergeMarkdownPages) -> dict[str, tp.Any]:
    _ = config
    page_texts: list[str] = json.loads(row["page_texts"])

    layout_regions: list[dict[str, tp.Any]] = []
    if row.get("layout_json"):
        try:
            layout_regions = json.loads(row["layout_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    all_articles: list[dict[str, tp.Any]] = []
    for crop_idx, page_text in enumerate(page_texts):
        real_page = crop_idx + 1
        if crop_idx < len(layout_regions):
            real_page = layout_regions[crop_idx].get("page", crop_idx + 1)
        for article in split_markdown_articles(page_text):
            article["page_span"] = [real_page]
            all_articles.append(article)

    return {**row, "result_json": json.dumps({"articles": all_articles})}
