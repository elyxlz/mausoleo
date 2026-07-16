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
    title_class_headlines: bool = False


def crop_page(layout_regions: list[tp.Any], crop_idx: int) -> int:
    if crop_idx < len(layout_regions) and isinstance(layout_regions[crop_idx], dict):
        return layout_regions[crop_idx].get("page", crop_idx + 1)
    return crop_idx + 1


def _split_raw_headline(text: str) -> tuple[str | None, str]:
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None, text
    first = lines[0].strip()
    if 3 <= len(first) <= 80 and not first.endswith((".", ",", ";", ":")):
        return first, "\n".join(lines[1:]).strip()
    return None, text


def _parse_crop_articles(page_text: str, raw_first_line_headline: bool) -> list[dict[str, tp.Any]]:
    page_data: tp.Any
    try:
        page_data = json.loads(_strip_markdown(page_text))
    except json.JSONDecodeError:
        headline, body = _split_raw_headline(page_text) if raw_first_line_headline else (None, page_text)
        fallback_article: dict[str, tp.Any] = {"unit_type": "article", "headline": headline, "paragraphs": [{"text": body}]}
        page_data = {"articles": [fallback_article]}

    if isinstance(page_data, dict):
        return page_data.get("articles", [])
    if isinstance(page_data, list):
        return page_data
    return []


def _crop_region(layout_regions: list[tp.Any], crop_idx: int) -> dict[str, tp.Any] | None:
    if crop_idx < len(layout_regions) and isinstance(layout_regions[crop_idx], dict):
        return layout_regions[crop_idx]
    return None


def _title_text(articles: list[dict[str, tp.Any]]) -> str | None:
    for art in articles:
        for para in art.get("paragraphs", []):
            for line in str(para.get("text", "")).splitlines():
                line = line.strip()
                if 3 <= len(line) <= 120:
                    return line
    return None


def _x_overlap(a: list[tp.Any], b: list[tp.Any]) -> float:
    return min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))


CropEntry = tuple[int, dict[str, tp.Any] | None, list[dict[str, tp.Any]]]


def _attach_title_headlines(entries: list[CropEntry]) -> None:
    for page, region, articles in entries:
        if not region or region.get("class") != "title":
            continue
        headline = _title_text(articles)
        if not headline:
            continue
        title_bbox = region.get("bbox", [0, 0, 0, 0])
        best: list[dict[str, tp.Any]] | None = None
        best_dy = float("inf")
        for other_page, other_region, other_articles in entries:
            if other_page != page or not other_region or other_region.get("class") == "title" or not other_articles:
                continue
            other_bbox = other_region.get("bbox", [0, 0, 0, 0])
            if _x_overlap(title_bbox, other_bbox) <= 0:
                continue
            dy = float(other_bbox[1]) - float(title_bbox[1])
            if -50 <= dy < best_dy:
                best_dy = dy
                best = other_articles
        if best is not None and not best[0].get("headline"):
            best[0]["headline"] = headline


@register_operator(MergePages, operation=OperatorType.MAP)
def merge_pages(row: dict[str, tp.Any], *, config: MergePages) -> dict[str, tp.Any]:
    page_texts: list[str] = json.loads(row["page_texts"])

    layout_regions: list[dict[str, tp.Any]] = []
    if "layout_json" in row and row["layout_json"]:
        try:
            layout_regions = json.loads(row["layout_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    entries: list[CropEntry] = []
    for crop_idx, page_text in enumerate(page_texts):
        real_page = crop_page(layout_regions, crop_idx)
        region = _crop_region(layout_regions, crop_idx)
        articles = _parse_crop_articles(page_text, config.raw_first_line_headline)
        entries.append((real_page, region, articles))

    if config.title_class_headlines:
        _attach_title_headlines(entries)

    all_articles: list[dict[str, tp.Any]] = []
    for real_page, region, articles in entries:
        if config.title_class_headlines and region is not None and region.get("class") == "title":
            continue
        for art in articles:
            art["page_span"] = [real_page]
            all_articles.append(art)

    return {**row, "result_json": json.dumps({"articles": all_articles})}
