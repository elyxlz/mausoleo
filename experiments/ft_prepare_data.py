from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
import sys
import typing as tp

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(pl.Path(__file__).parent))

from grouper_features import DATES, align_regions_to_gt, features, load_gt, load_regions, start_labels, words

REPO = pl.Path(__file__).resolve().parents[1]
GT_DIR = REPO / "eval" / "ground_truth"
OUT_DIR = REPO / "eval" / "autoresearch" / "paddleft"
PROMPT = "OCR:"
MIN_JACCARD = 0.5
VAL_EVERY = 10
MAX_SIDE = 1600
CROP_PAD = 12


@dc.dataclass(frozen=True)
class Pair:
    date: str
    group_index: int
    image_path: str
    label: str
    jaccard: float


def train_grouper(dates: list[str], exclude: str) -> GradientBoostingClassifier:
    X: list[list[float]] = []
    y: list[int] = []
    for d in dates:
        if d == exclude:
            continue
        rg = load_regions(d)
        X.extend(features(rg))
        y.extend(start_labels(rg, align_regions_to_gt(rg, load_gt(d))))
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05).fit(np.array(X), np.array(y))


def group_indices(starts: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, s in enumerate(starts):
        if s and cur:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def predict_groups(regions: list[dict[str, tp.Any]], clf: GradientBoostingClassifier) -> list[list[int]]:
    starts: list[int] = np.asarray(clf.predict(np.array(features(regions)))).tolist()
    if starts:
        starts[0] = 1
    return group_indices(starts)


def union_bbox(boxes: list[list[int]], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(b[0] for b in boxes) - CROP_PAD)
    y1 = max(0, min(b[1] for b in boxes) - CROP_PAD)
    x2 = min(width, max(b[2] for b in boxes) + CROP_PAD)
    y2 = min(height, max(b[3] for b in boxes) + CROP_PAD)
    return x1, y1, x2, y2


def resize_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return img
    s = MAX_SIDE / max(w, h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


def gt_full_text(article: dict[str, tp.Any]) -> str:
    body = "\n".join(p["text"] for p in article["paragraphs"])
    headline = article.get("headline")
    return f"{headline}\n{body}" if headline else body


def best_gt_match(proxy_text: str, gt_texts: list[str]) -> tuple[int, float]:
    pw = words(proxy_text)
    if len(pw) < 5:
        return -1, 0.0
    best_i, best_j = -1, 0.0
    for gi, gt in enumerate(gt_texts):
        gw = words(gt)
        if not gw:
            continue
        j = len(pw & gw) / len(pw | gw)
        if j > best_j:
            best_j, best_i = j, gi
    return best_i, best_j


def load_page_images(date: str) -> dict[int, Image.Image]:
    pages = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    return {int(p.stem): Image.open(p).convert("RGB") for p in pages}


def extract_candidates(date: str, train_dates: list[str]) -> list[tuple[int, Image.Image, str]]:
    regions = load_regions(date)
    groups = predict_groups(regions, train_grouper(train_dates, exclude=date))
    page_images = load_page_images(date)
    candidates: list[tuple[int, Image.Image, str]] = []
    for gi, grp in enumerate(groups):
        pages = {regions[i]["page"] for i in grp}
        if len(pages) != 1:
            continue
        img = page_images[pages.pop()]
        x1, y1, x2, y2 = union_bbox([regions[i]["bbox"] for i in grp], img.width, img.height)
        proxy = "\n".join(regions[i]["text"].strip() for i in grp if regions[i]["text"].strip())
        candidates.append((gi, resize_crop(img.crop((x1, y1, x2, y2))), proxy))
    return candidates


def match_date(date: str, train_dates: list[str], crop_dir: pl.Path) -> list[Pair]:
    gt_texts = [gt_full_text(a) for a in load_gt(date)]
    best_by_gt: dict[int, tuple[float, int, Image.Image]] = {}
    for gi, crop, proxy in extract_candidates(date, train_dates):
        match_i, jac = best_gt_match(proxy, gt_texts)
        if match_i < 0 or jac < MIN_JACCARD:
            continue
        if match_i not in best_by_gt or jac > best_by_gt[match_i][0]:
            best_by_gt[match_i] = (jac, gi, crop)
    pairs: list[Pair] = []
    for match_i, (jac, gi, crop) in sorted(best_by_gt.items()):
        path = crop_dir / f"{date}_g{gi:03d}.jpg"
        crop.save(path, quality=92)
        pairs.append(Pair(date, gi, str(path), gt_texts[match_i], round(jac, 4)))
    return pairs


def build_record(pair: Pair) -> dict[str, tp.Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": pair.label},
        ],
        "images": [pair.image_path],
    }


def write_jsonl(path: pl.Path, pairs: list[Pair]) -> None:
    path.write_text("\n".join(json.dumps(build_record(p), ensure_ascii=False) for p in pairs) + "\n")


def print_stats(test_date: str, train_dates: list[str], train: list[Pair], val: list[Pair]) -> None:
    print(f"test issue (excluded): {test_date}")
    print(f"train issues: {train_dates}")
    all_pairs = train + val
    for d in train_dates:
        dp = [p for p in all_pairs if p.date == d]
        jacs = [p.jaccard for p in dp]
        print(f"  {d}: {len(dp)} pairs, mean jaccard {np.mean(jacs):.3f}" if dp else f"  {d}: 0 pairs")
    word_counts = [len(p.label.split()) for p in all_pairs]
    print(f"total: {len(all_pairs)} pairs ({len(train)} train / {len(val)} val)")
    print(f"label words: mean {np.mean(word_counts):.0f}, p95 {np.percentile(word_counts, 95):.0f}, max {max(word_counts)}")
    for p in all_pairs[:2]:
        print(f"--- example {p.date} g{p.group_index} jaccard={p.jaccard} image={p.image_path}")
        print(p.label[:300].replace("\n", " | "))


def main() -> None:
    test_date = sys.argv[1] if len(sys.argv) > 1 else "1935-06-15"
    assert test_date in DATES
    train_dates = [d for d in DATES if d != test_date]
    crop_dir = OUT_DIR / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[Pair] = []
    for d in train_dates:
        pairs.extend(match_date(d, train_dates, crop_dir))
    val = [p for i, p in enumerate(pairs) if i % VAL_EVERY == 0]
    train = [p for i, p in enumerate(pairs) if i % VAL_EVERY != 0]
    tag = f"no{test_date}"
    write_jsonl(OUT_DIR / f"train_{tag}.jsonl", train)
    write_jsonl(OUT_DIR / f"val_{tag}.jsonl", val)
    print_stats(test_date, train_dates, train, val)


if __name__ == "__main__":
    main()
