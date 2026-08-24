from __future__ import annotations

import collections
import json
import re
import typing as tp

from mausoleo.eval.evaluate import article_text
from mausoleo.paths import EVAL_DATES, GT_DIR

LEXICON_DATES = EVAL_DATES
WORD_RE = re.compile(r"[a-zàèéìòóù]{2,}")


def build_lexicon() -> frozenset[str]:
    words: set[str] = set()
    for date in LEXICON_DATES:
        gt_path = GT_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            continue
        for article in json.loads(gt_path.read_text()).get("articles", []):
            for paragraph in article.get("paragraphs", []):
                words.update(WORD_RE.findall(paragraph.get("text", "").lower()))
    return frozenset(words)


def lexicon_validity(text: str, lexicon: frozenset[str]) -> float:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in lexicon) / len(tokens)


def repetition_rate(text: str) -> float:
    tokens = text.lower().split()
    if len(tokens) < 20:
        return 0.0
    trigrams = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    most_common = collections.Counter(trigrams).most_common(1)[0][1]
    return most_common / len(trigrams)


def probe_issue(pred_issue: dict[str, tp.Any], lexicon: frozenset[str]) -> dict[str, float]:
    articles = pred_issue.get("articles", [])
    texts = [article_text(a) for a in articles]
    lengths = sorted(len(t) for t in texts)
    n = len(texts)
    if n == 0:
        return {"articles": 0.0}
    full_text = " ".join(texts)
    return {
        "articles": float(n),
        "lexicon_validity": lexicon_validity(full_text, lexicon),
        "mean_repetition": sum(repetition_rate(t) for t in texts) / n,
        "median_chars": float(lengths[n // 2]),
        "total_chars": float(sum(lengths)),
        "share_tiny_lt50": sum(1 for length in lengths if length < 50) / n,
        "share_huge_gt5k": sum(1 for length in lengths if length > 5000) / n,
        "share_high_repetition": sum(1 for t in texts if repetition_rate(t) > 0.10) / n,
    }
