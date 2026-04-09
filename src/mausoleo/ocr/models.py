from __future__ import annotations

import dataclasses as dc
import typing as tp


@dc.dataclass(frozen=True)
class Paragraph:
    id: str
    text: str


@dc.dataclass(frozen=True)
class Article:
    id: str
    unit_type: str
    headline: str | None
    paragraphs: list[Paragraph]
    page_span: list[int]
    position_in_issue: int


@dc.dataclass(frozen=True)
class Issue:
    date: str
    source: str
    page_count: int
    articles: list[Article]
