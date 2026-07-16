from __future__ import annotations

import dataclasses as dc
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


@dc.dataclass(frozen=True, kw_only=True)
class MergePages(BaseOperatorConfig):
    raw_first_line_headline: bool = False


def _split_raw_headline(text: str) -> tuple[str | None, str]:
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None, text
    first = lines[0].strip()
    if 3 <= len(first) <= 80 and not first.endswith((".", ",", ";", ":")):
        return first, "\n".join(lines[1:]).strip()
    return None, text


@register_operator(MergePages, operation=OperatorType.MAP)
def merge_pages(row: dict[str, tp.Any], *, config: MergePages) -> dict[str, tp.Any]:
    page_texts: list[str] = json.loads(row["page_texts"])

    layout_regions: list[dict[str, tp.Any]] = []
    if "layout_json" in row and row["layout_json"]:
        try:
            layout_regions = json.loads(row["layout_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    all_articles: list[dict[str, tp.Any]] = []
    for crop_idx, page_text in enumerate(page_texts):
        real_page = crop_idx + 1
        if crop_idx < len(layout_regions):
            real_page = layout_regions[crop_idx].get("page", crop_idx + 1)

        page_data: tp.Any
        try:
            page_data = json.loads(_strip_markdown(page_text))
        except json.JSONDecodeError:
            headline, body = _split_raw_headline(page_text) if config.raw_first_line_headline else (None, page_text)
            fallback_article: dict[str, tp.Any] = {"unit_type": "article", "headline": headline, "paragraphs": [{"text": body}]}
            page_data = {"articles": [fallback_article]}

        articles: list[dict[str, tp.Any]] = []
        if isinstance(page_data, dict):
            articles = page_data.get("articles", [])
        elif isinstance(page_data, list):
            articles = page_data

        for art in articles:
            art["page_span"] = [real_page]
            all_articles.append(art)

    return {**row, "result_json": json.dumps({"articles": all_articles})}
