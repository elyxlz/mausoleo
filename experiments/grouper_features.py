from __future__ import annotations

import json
import pathlib as pl
import re
import typing as tp

WORK_DIR = pl.Path("eval/autoresearch/semgroup")
GT_DIR = pl.Path("eval/ground_truth")
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

_WORD = re.compile(r"[a-zàèéìòóù0-9]{2,}")
_DATELINE = re.compile(r"^[A-ZÀ-Ù][A-Za-zÀ-ù.]+,?\s*\d|\(per telegr|\(nostro", re.I)


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def load_regions(date: str) -> list[dict[str, tp.Any]]:
    return json.loads((WORK_DIR / f"regions_{date}.json").read_text())


def load_gt(date: str) -> list[dict[str, tp.Any]]:
    return json.loads((GT_DIR / date / "ground_truth.json").read_text())["articles"]


def align_regions_to_gt(regions: list[dict[str, tp.Any]], gt: list[dict[str, tp.Any]]) -> list[int]:
    """Assign each region to the GT article index whose text best contains it; -1 if none."""
    gt_words = [words(" ".join(p["text"] for p in a["paragraphs"]) + " " + (a.get("headline") or "")) for a in gt]
    assigned: list[int] = []
    for r in regions:
        rw = words(r["text"])
        if len(rw) < 3:
            assigned.append(-1)
            continue
        best_i, best_c = -1, 0.0
        for gi, gw in enumerate(gt_words):
            if not gw:
                continue
            c = len(rw & gw) / len(rw)
            if c > best_c:
                best_c, best_i = c, gi
        assigned.append(best_i if best_c >= 0.45 else -1)
    return assigned


def start_labels(regions: list[dict[str, tp.Any]], assigned: list[int]) -> list[int]:
    """1 if region begins a new article in reading order, else 0."""
    labels = []
    for i, a in enumerate(assigned):
        if i == 0:
            labels.append(1)
        elif a == -1 or assigned[i - 1] == -1:
            labels.append(1)
        else:
            labels.append(1 if a != assigned[i - 1] else 0)
    return labels


def _page_dims(regions: list[dict[str, tp.Any]]) -> dict[int, tuple[int, int]]:
    dims: dict[int, tuple[int, int]] = {}
    for r in regions:
        w, h = dims.get(r["page"], (0, 0))
        dims[r["page"]] = (max(w, r["bbox"][2]), max(h, r["bbox"][3]))
    return dims


FEATURE_NAMES = [
    "is_title",
    "prev_is_title",
    "page_changed",
    "y_gap_norm",
    "x_overlap",
    "same_column",
    "log_len",
    "upper_ratio",
    "has_dateline",
    "is_page_top",
    "col_index",
    "col_frac",
    "n_cols",
    "prev_short",
]


def page_column_edges(regions: list[dict[str, tp.Any]], page: int, width: int) -> list[float]:
    spans = sorted((r["bbox"][0], r["bbox"][2]) for r in regions if r["page"] == page)
    if not spans:
        return [0.0, float(width)]
    widths = sorted(x2 - x1 for x1, x2 in spans)
    median = widths[len(widths) // 2] or width
    narrow = [(x1, x2) for x1, x2 in spans if (x2 - x1) <= median * 1.7] or spans
    merged: list[list[float]] = []
    for x1, x2 in narrow:
        if merged and x1 < merged[-1][1] - median * 0.25:
            merged[-1][1] = max(merged[-1][1], x2)
        else:
            merged.append([float(x1), float(x2)])
    bands = [b for b in merged if b[1] - b[0] >= width * 0.03] or [[0.0, float(width)]]
    return [b[0] for b in bands] + [bands[-1][1]]


def column_of(edges: list[float], x1: float) -> float:
    for index in range(len(edges) - 1):
        if x1 < edges[index + 1]:
            return float(index)
    return float(max(0, len(edges) - 2))


def features(regions: list[dict[str, tp.Any]]) -> list[list[float]]:
    dims = _page_dims(regions)
    edges = {page: page_column_edges(regions, page, max(dims[page][0], 1)) for page in {r["page"] for r in regions}}
    feats: list[list[float]] = []
    prev = None
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        pw, ph = dims[r["page"]]
        pw, ph = max(pw, 1), max(ph, 1)
        is_title = 1.0 if r["class"] == "title" else 0.0
        txt = r["text"].strip()
        letters = [c for c in txt if c.isalpha()]
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        has_dateline = 1.0 if _DATELINE.search(txt) else 0.0
        col_index = column_of(edges[r["page"]], x1)
        n_cols = float(len(edges[r["page"]]) - 1)
        if prev is None or prev["page"] != r["page"]:
            page_changed = 1.0 if prev is not None else 0.0
            y_gap, x_ov, same_col, prev_title, prev_short = 1.0, 0.0, 0.0, 0.0, 0.0
        else:
            px1, py1, px2, py2 = prev["bbox"]
            page_changed = 0.0
            y_gap = max(-0.2, min(1.0, (y1 - py2) / ph))
            ov = min(x2, px2) - max(x1, px1)
            x_ov = max(0.0, ov) / max(1, min(x2 - x1, px2 - px1))
            same_col = 1.0 if x_ov >= 0.5 else 0.0
            prev_title = 1.0 if prev["class"] == "title" else 0.0
            prev_short = 1.0 if len(prev["text"].strip()) < 40 else 0.0
        import math

        feats.append(
            [
                is_title,
                prev_title,
                page_changed,
                y_gap,
                x_ov,
                same_col,
                math.log1p(len(txt)),
                upper_ratio,
                has_dateline,
                1.0 if (prev is None or prev["page"] != r["page"]) else 0.0,
                col_index,
                col_index / max(n_cols, 1.0),
                n_cols,
                prev_short,
            ]
        )
        prev = r
    return feats


def build_dataset(dates: list[str] = DATES) -> tuple[list[list[float]], list[int], list[str]]:
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    for date in dates:
        regions = load_regions(date)
        gt = load_gt(date)
        assigned = align_regions_to_gt(regions, gt)
        labels = start_labels(regions, assigned)
        X.extend(features(regions))
        y.extend(labels)
        groups.extend([date] * len(regions))
    return X, y, groups
