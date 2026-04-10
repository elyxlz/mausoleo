from __future__ import annotations

import dataclasses as dc
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class ParseIssue(BaseOperatorConfig):
    pass


def _extract_json(text: str) -> tp.Any:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return _repair_truncated_json(text)


def _repair_truncated_json(text: str) -> tp.Any:
    closers = [
        '"}]}', '"} ]}', '"}]}', '"}]  }', '"}]}',
        '"}, ]}', '"}  ]}', '"}]}',
        '"}]', '"]', '"}', '}', ']',
        '"} ]  }', '"}  ]  }',
    ]
    for suffix in closers:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    for n_close in range(1, 6):
        for combo in _brace_combos(n_close):
            try:
                return json.loads(text + combo)
            except json.JSONDecodeError:
                continue

    return {"articles": [{"unit_type": "article", "headline": None, "paragraphs": [{"text": text}]}]}


def _brace_combos(n: int) -> list[str]:
    if n == 1:
        return ["}", "]", '"}', '"]']
    results: list[str] = []
    for prefix in _brace_combos(1):
        for suffix in _brace_combos(n - 1):
            results.append(prefix + suffix)
    return results


def _build_issue_json(raw: str, date: str, source: str, page_count: int) -> str:
    parsed = _extract_json(raw)
    articles_data: list[tp.Any] = parsed.get("articles", []) if isinstance(parsed, dict) else parsed

    articles = []
    for i, art in enumerate(articles_data):
        raw_paragraphs = art.get("paragraphs", [])
        if not raw_paragraphs and "text" in art:
            raw_paragraphs = [{"text": art["text"]}]

        paragraphs = []
        for j, para in enumerate(raw_paragraphs):
            para_text = para.get("text", str(para)) if hasattr(para, "get") else str(para)
            paragraphs.append({"id": f"{date}_a{i:02d}_p{j:02d}", "text": para_text})

        articles.append(
            {
                "id": f"{date}_a{i:02d}",
                "unit_type": art.get("unit_type", "article"),
                "headline": art.get("headline"),
                "paragraphs": paragraphs,
                "page_span": art.get("page_span", []),
                "position_in_issue": i,
            }
        )

    issue = {"date": date, "source": source, "page_count": page_count, "articles": articles}
    return json.dumps(issue)


@register_operator(ParseIssue, operation=OperatorType.MAP)
def parse_issue(row: dict[str, tp.Any], *, config: ParseIssue) -> dict[str, tp.Any]:
    _ = config
    raw = row["result_json"]
    issue_json = _build_issue_json(
        raw=raw,
        date=row["date"],
        source=row["source"],
        page_count=row["page_count"],
    )
    return {**row, "issue_json": issue_json}
