from __future__ import annotations

import dataclasses as dc
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class EnsembleMerge(BaseOperatorConfig):
    overlap_threshold: float = 0.30
    min_article_chars: int = 30


def _normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text).strip().lower()


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _word_overlap(a: str, b: str) -> float:
    a_words = set(_normalize(a).split())
    b_words = set(_normalize(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


@register_operator(EnsembleMerge, operation=OperatorType.MAP)
def ensemble_merge(row: dict[str, tp.Any], *, config: EnsembleMerge) -> dict[str, tp.Any]:
    if config.mock:
        return row

    primary_json = row.get("primary_issue_json", "")
    secondary_json = row.get("secondary_issue_json", "")

    if not primary_json or not secondary_json:
        return row

    primary = json.loads(primary_json)
    secondary = json.loads(secondary_json)

    pa = primary.get("articles", [])
    sa = secondary.get("articles", [])

    pt = [_normalize(_article_text(a)) for a in pa]

    used: set[int] = set()
    for p in pt:
        for si, s in enumerate(sa):
            if si in used:
                continue
            if _word_overlap(p, _normalize(_article_text(s))) >= config.overlap_threshold:
                used.add(si)

    new = [
        sa[si]
        for si in range(len(sa))
        if si not in used and len(_article_text(sa[si]).strip()) >= config.min_article_chars
    ]

    merged = list(pa) + new
    indexed = list(enumerate(merged))
    indexed.sort(key=lambda x: (x[1].get("page_span", [999])[0] if x[1].get("page_span") else 999, x[0]))
    primary["articles"] = [a for _, a in indexed]

    result = dict(row)
    result["issue_json"] = json.dumps(primary)
    return result
