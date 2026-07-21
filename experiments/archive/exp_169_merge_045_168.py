from __future__ import annotations

import json
import pathlib as pl
import sys

sys.path.insert(0, "src")

from mausoleo.ocr.merge import article_text, normalize_text, quality_score, text_overlap

PRED_DIR = pl.Path("eval/predictions")
PRIMARY = "exp_045_qwen3vl_vllm"
SECONDARY = "exp_168_grouped_qwen"
OVERLAP_THRESHOLD = 0.50
MIN_ARTICLE_CHARS = 30
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]


def _load(config: str, date: str) -> dict:
    return json.loads((PRED_DIR / f"{config}_{date}.json").read_text())


def merge_quality(primary: dict, secondary: dict) -> dict:
    prim = [dict(a) for a in primary.get("articles", [])]
    sec = secondary.get("articles", [])
    p_norm = [normalize_text(article_text(a)) for a in prim]
    s_norm = [normalize_text(article_text(a)) for a in sec]
    used: set[int] = set()
    for pi, pa in enumerate(prim):
        best_si, best_ov = -1, 0.0
        for si in range(len(sec)):
            if si in used:
                continue
            ov = text_overlap(p_norm[pi], s_norm[si])
            if ov >= OVERLAP_THRESHOLD and ov > best_ov:
                best_ov, best_si = ov, si
        if best_si >= 0:
            used.add(best_si)
            sa = sec[best_si]
            if quality_score(article_text(sa)) > quality_score(article_text(pa)):
                merged = dict(sa)
                merged["page_span"] = pa.get("page_span", sa.get("page_span"))
                merged["headline"] = sa.get("headline") or pa.get("headline")
                prim[pi] = merged
    new = [sa for si, sa in enumerate(sec) if si not in used and len(article_text(sa).strip()) >= MIN_ARTICLE_CHARS]
    out = dict(primary)
    out["source"] = "exp_169_merge_045_168"
    out["articles"] = prim + new
    return out


def main() -> None:
    for date in (sys.argv[1:] or DATES):
        pred = merge_quality(_load(PRIMARY, date), _load(SECONDARY, date))
        out = PRED_DIR / f"exp_169_merge_045_168_{date}.json"
        out.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles -> {out}")


if __name__ == "__main__":
    main()
