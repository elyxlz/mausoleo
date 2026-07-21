from __future__ import annotations

import json
import pathlib as pl
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(pl.Path(__file__).parent))
sys.path.insert(0, "src")

from grouper_features import DATES, features, load_regions, load_gt, align_regions_to_gt, start_labels
from mausoleo.ocr.merge import trim_trailing_garbage
from mausoleo.ocr.operators.merge import _strip_markdown, squeeze_char_runs

PRED_DIR = pl.Path("eval/predictions")


def clean_text(text: str) -> str:
    return trim_trailing_garbage(squeeze_char_runs(_strip_markdown(text))).strip()


def _train_dataset(exclude: str) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[float]] = []
    y: list[int] = []
    for date in DATES:
        if date == exclude:
            continue
        regions = load_regions(date)
        X.extend(features(regions))
        y.extend(start_labels(regions, align_regions_to_gt(regions, load_gt(date))))
    return np.array(X), np.array(y)


def _group(regions: list[dict], starts: list[int]) -> list[dict]:
    articles: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        title = next((clean_text(r["text"]) for r in cur if r["class"] == "title" and clean_text(r["text"])), None)
        body = [r for r in cur if not (r["class"] == "title" and clean_text(r["text"]) == title)]
        text = "\n".join(clean_text(r["text"]) for r in (body or cur) if clean_text(r["text"]))
        pages = sorted({r["page"] for r in cur})
        articles.append({"unit_type": "article", "headline": title,
                         "paragraphs": [{"text": text}],
                         "page_span": [pages[0], pages[-1]] if len(pages) > 1 else [pages[0]]})

    for r, s in zip(regions, starts):
        if s and cur:
            flush()
            cur = []
        cur.append(r)
    flush()
    return articles


def predict_issue(date: str) -> dict:
    X_tr, y_tr = _train_dataset(exclude=date)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05).fit(X_tr, y_tr)
    regions = load_regions(date)
    starts = clf.predict(np.array(features(regions))).tolist()
    if regions:
        starts[0] = 1
    return {"date": date, "source": "exp_171_grouped_clean", "articles": _group(regions, starts)}


def main() -> None:
    for date in (sys.argv[1:] or list(DATES)):
        pred = predict_issue(date)
        out = PRED_DIR / f"exp_171_grouped_clean_{date}.json"
        out.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles -> {out}")


if __name__ == "__main__":
    main()
