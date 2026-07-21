from __future__ import annotations

import json
import pathlib as pl
import sys

sys.path.insert(0, "src")

from mausoleo.ocr.merge import article_text, normalize_text, quality_score, text_overlap

PRED_DIR = pl.Path("eval/predictions")
BASE = "exp_168_grouped_qwen"
TEXT_SOURCE = "exp_045_qwen3vl_vllm"
OVERLAP_THRESHOLD = 0.50
BLOB_LEN_RATIO = 2.0
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]


def _load(config: str, date: str) -> dict:
    return json.loads((PRED_DIR / f"{config}_{date}.json").read_text())


def asym_merge(base: dict, text_src: dict) -> dict:
    base_arts = [dict(a) for a in base.get("articles", [])]
    src_arts = text_src.get("articles", [])
    b_norm = [normalize_text(article_text(a)) for a in base_arts]
    s_norm = [normalize_text(article_text(a)) for a in src_arts]
    for bi, ba in enumerate(base_arts):
        b_len = len(article_text(ba))
        best_si, best_ov = -1, 0.0
        for si in range(len(src_arts)):
            ov = text_overlap(b_norm[bi], s_norm[si])
            if ov >= OVERLAP_THRESHOLD and ov > best_ov:
                best_ov, best_si = ov, si
        if best_si < 0:
            continue
        sa = src_arts[best_si]
        s_len = len(article_text(sa))
        if s_len > b_len * BLOB_LEN_RATIO:
            continue
        if quality_score(article_text(sa)) > quality_score(article_text(ba)):
            ba["paragraphs"] = sa.get("paragraphs", ba["paragraphs"])
            ba["headline"] = ba.get("headline") or sa.get("headline")
    out = dict(base)
    out["source"] = "exp_170_asym_merge"
    out["articles"] = base_arts
    return out


def main() -> None:
    for date in (sys.argv[1:] or DATES):
        pred = asym_merge(_load(BASE, date), _load(TEXT_SOURCE, date))
        out = PRED_DIR / f"exp_170_asym_merge_{date}.json"
        out.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles -> {out}")


if __name__ == "__main__":
    main()
