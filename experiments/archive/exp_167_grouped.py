from __future__ import annotations

import json
import pathlib as pl
import sys
import typing as tp

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(pl.Path(__file__).parent))

from grouper_features import DATES, features, load_regions, load_gt, align_regions_to_gt, start_labels

PRED_DIR = pl.Path("eval/predictions")


def _train_dataset(exclude: str) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[float]] = []
    y: list[int] = []
    for date in DATES:
        if date == exclude:
            continue
        regions = load_regions(date)
        assigned = align_regions_to_gt(regions, load_gt(date))
        X.extend(features(regions))
        y.extend(start_labels(regions, assigned))
    return np.array(X), np.array(y)


def _group_regions(regions: list[dict[str, tp.Any]], starts: list[int]) -> list[dict[str, tp.Any]]:
    articles: list[dict[str, tp.Any]] = []
    cur: list[dict[str, tp.Any]] = []

    def flush() -> None:
        if not cur:
            return
        title = next((r["text"].strip() for r in cur if r["class"] == "title" and r["text"].strip()), None)
        body = [r for r in cur if not (r["class"] == "title" and r["text"].strip() == title)]
        text = "\n".join(r["text"].strip() for r in (body or cur) if r["text"].strip())
        pages = sorted({r["page"] for r in cur})
        articles.append(
            {
                "unit_type": "article",
                "headline": title,
                "paragraphs": [{"text": text}],
                "page_span": [pages[0], pages[-1]] if len(pages) > 1 else [pages[0]],
            }
        )

    for r, s in zip(regions, starts):
        if s and cur:
            flush()
            cur = []
        cur.append(r)
    flush()
    return articles


def predict_issue(date: str) -> dict[str, tp.Any]:
    X_tr, y_tr = _train_dataset(exclude=date)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
    clf.fit(X_tr, y_tr)
    regions = load_regions(date)
    starts = clf.predict(np.array(features(regions))).tolist()
    if regions:
        starts[0] = 1
    return {"date": date, "articles": _group_regions(regions, starts)}


def main() -> None:
    dates = sys.argv[1:] or list(DATES)
    for date in dates:
        pred = predict_issue(date)
        out = PRED_DIR / f"exp_167_grouped_{date}.json"
        out.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles -> {out}")


if __name__ == "__main__":
    main()
