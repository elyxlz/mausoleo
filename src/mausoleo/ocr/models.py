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


def issue_from_dict(data: dict[str, tp.Any]) -> Issue:
    return Issue(
        date=data["date"],
        source=data["source"],
        page_count=data["page_count"],
        articles=[
            Article(
                id=a["id"],
                unit_type=a["unit_type"],
                headline=a.get("headline"),
                paragraphs=[Paragraph(id=p["id"], text=p.get("text") or "") for p in a["paragraphs"]],
                page_span=a["page_span"],
                position_in_issue=a["position_in_issue"],
            )
            for a in data["articles"]
        ],
    )


def extract_full_text(issue: Issue) -> str:
    parts: list[str] = []
    for article in issue.articles:
        if article.headline:
            parts.append(article.headline)
        for para in article.paragraphs:
            if para.text:
                parts.append(para.text)
    return "\n\n".join(parts)
